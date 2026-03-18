classdef NanoUdpBatchProtocol
    %NANOUDPBATCHPROTOCOL Fixed-layout Nano 33 IoT UDP batch packets.
    methods (Static)
        function protocolName = defaultProtocol()
            protocolName = "udp_batch_binary_v1";
        end

        function payloadTemplates = buildCommandTemplates(sampleSequence, activeMask, pulseUsBySample)
            sampleSequence = reshape(uint32(sampleSequence), [], 1);
            pulseUsBySample = reshape(uint16(pulseUsBySample), [], 4);
            sampleCount = numel(sampleSequence);
            payloadTemplates = zeros(NanoUdpBatchProtocol.commandPayloadLengthBytes(), sampleCount, "uint8");

            for sampleIndex = 1:sampleCount
                payloadTemplates(:, sampleIndex) = NanoUdpBatchProtocol.buildCommandTemplate( ...
                    sampleSequence(sampleIndex), ...
                    activeMask, ...
                    pulseUsBySample(sampleIndex, :));
            end
        end

        function payloadBytes = buildCommandTemplate(sampleSequence, activeMask, pulseUs)
            payloadBytes = zeros(NanoUdpBatchProtocol.commandPayloadLengthBytes(), 1, "uint8");
            payloadBytes(1:4) = NanoUdpBatchProtocol.commandMagicBytes();
            payloadBytes(5:8) = reshape(typecast(uint32(sampleSequence), "uint8"), [], 1);
            payloadBytes(13) = uint8(activeMask);
            payloadBytes(15:22) = reshape(typecast(reshape(uint16(pulseUs), 1, 4), "uint8"), [], 1);
        end

        function payloadBytes = setCommandHostTxUs(payloadTemplate, hostTxUs)
            payloadBytes = reshape(uint8(payloadTemplate), [], 1);
            payloadBytes(9:12) = reshape(typecast(uint32(hostTxUs), "uint8"), [], 1);
        end

        function payloadBytes = buildTelemetryPayload(sampleSequence, hostTxUs, boardRxUs, applyStartUs, applyEndUs, activeMask, statusCode)
            payloadBytes = zeros(NanoUdpBatchProtocol.telemetryPayloadLengthBytes(), 1, "uint8");
            payloadBytes(1:4) = NanoUdpBatchProtocol.telemetryMagicBytes();
            payloadBytes(5:8) = reshape(typecast(uint32(sampleSequence), "uint8"), [], 1);
            payloadBytes(9:12) = reshape(typecast(uint32(hostTxUs), "uint8"), [], 1);
            payloadBytes(13:16) = reshape(typecast(uint32(boardRxUs), "uint8"), [], 1);
            payloadBytes(17:20) = reshape(typecast(uint32(applyStartUs), "uint8"), [], 1);
            payloadBytes(21:24) = reshape(typecast(uint32(applyEndUs), "uint8"), [], 1);
            payloadBytes(25) = uint8(activeMask);
            payloadBytes(26) = uint8(statusCode);
        end

        function decodedPayload = decodeCommandPayload(payloadBytes)
            payloadBytes = reshape(uint8(payloadBytes), [], 1);
            decodedPayload = struct( ...
                "sampleSequence", uint32(0), ...
                "hostTxUs", uint32(0), ...
                "activeMask", uint8(0), ...
                "pulseUs", zeros(1, 4, "uint16"));

            if numel(payloadBytes) ~= NanoUdpBatchProtocol.commandPayloadLengthBytes() || ...
                    ~isequal(payloadBytes(1:4), NanoUdpBatchProtocol.commandMagicBytes())
                error("NanoUdpBatchProtocol:InvalidCommandPayload", ...
                    "Command payload must be a %d-byte packet starting with '%s'.", ...
                    NanoUdpBatchProtocol.commandPayloadLengthBytes(), ...
                    char(NanoUdpBatchProtocol.commandMagicBytes().'));
            end

            decodedPayload.sampleSequence = typecast(payloadBytes(5:8), "uint32");
            decodedPayload.hostTxUs = typecast(payloadBytes(9:12), "uint32");
            decodedPayload.activeMask = payloadBytes(13);
            decodedPayload.pulseUs = reshape(typecast(payloadBytes(15:22), "uint16"), 1, 4);
        end

        function decodedPayload = decodeTelemetryPayload(payloadBytes)
            payloadBytes = reshape(uint8(payloadBytes), [], 1);
            decodedPayload = struct( ...
                "sampleSequence", uint32(0), ...
                "hostTxUs", uint32(0), ...
                "boardRxUs", uint32(0), ...
                "applyStartUs", uint32(0), ...
                "applyEndUs", uint32(0), ...
                "activeMask", uint8(0), ...
                "statusCode", uint8(0));

            if numel(payloadBytes) ~= NanoUdpBatchProtocol.telemetryPayloadLengthBytes() || ...
                    ~isequal(payloadBytes(1:4), NanoUdpBatchProtocol.telemetryMagicBytes())
                error("NanoUdpBatchProtocol:InvalidTelemetryPayload", ...
                    "Telemetry payload must be a %d-byte packet starting with '%s'.", ...
                    NanoUdpBatchProtocol.telemetryPayloadLengthBytes(), ...
                    char(NanoUdpBatchProtocol.telemetryMagicBytes().'));
            end

            decodedPayload.sampleSequence = typecast(payloadBytes(5:8), "uint32");
            decodedPayload.hostTxUs = typecast(payloadBytes(9:12), "uint32");
            decodedPayload.boardRxUs = typecast(payloadBytes(13:16), "uint32");
            decodedPayload.applyStartUs = typecast(payloadBytes(17:20), "uint32");
            decodedPayload.applyEndUs = typecast(payloadBytes(21:24), "uint32");
            decodedPayload.activeMask = payloadBytes(25);
            decodedPayload.statusCode = payloadBytes(26);
        end

        function isMatch = isCommandPayload(payloadBytes)
            payloadBytes = reshape(uint8(payloadBytes), [], 1);
            isMatch = numel(payloadBytes) == NanoUdpBatchProtocol.commandPayloadLengthBytes() && ...
                isequal(payloadBytes(1:4), NanoUdpBatchProtocol.commandMagicBytes());
        end

        function isMatch = isTelemetryPayload(payloadBytes)
            payloadBytes = reshape(uint8(payloadBytes), [], 1);
            isMatch = numel(payloadBytes) == NanoUdpBatchProtocol.telemetryPayloadLengthBytes() && ...
                isequal(payloadBytes(1:4), NanoUdpBatchProtocol.telemetryMagicBytes());
        end

        function payloadLine = formatPayloadForTextLog(payloadBytes)
            payloadBytes = reshape(uint8(payloadBytes), 1, []);

            if NanoUdpBatchProtocol.isTelemetryPayload(payloadBytes)
                telemetryPayload = NanoUdpBatchProtocol.decodeTelemetryPayload(payloadBytes);
                payloadLine = compose( ...
                    "BATCH_EVENT,%u,%u,%u,%u,%u,%u,%u", ...
                    double(telemetryPayload.sampleSequence), ...
                    double(telemetryPayload.hostTxUs), ...
                    double(telemetryPayload.boardRxUs), ...
                    double(telemetryPayload.applyStartUs), ...
                    double(telemetryPayload.applyEndUs), ...
                    double(telemetryPayload.activeMask), ...
                    double(telemetryPayload.statusCode));
                return;
            end

            payloadLine = string(strtrim(char(payloadBytes)));
        end

        function payloadLengthBytes = commandPayloadLengthBytes()
            payloadLengthBytes = 24;
        end

        function payloadLengthBytes = telemetryPayloadLengthBytes()
            payloadLengthBytes = 28;
        end

        function magicBytes = commandMagicBytes()
            magicBytes = reshape(uint8('N3C1'), [], 1);
        end

        function magicBytes = telemetryMagicBytes()
            magicBytes = reshape(uint8('N3E1'), [], 1);
        end
    end
end
