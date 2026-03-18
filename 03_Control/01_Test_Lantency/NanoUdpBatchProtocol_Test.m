function tests = NanoUdpBatchProtocol_Test
%NANOUDPBATCHPROTOCOL_TEST Unit tests for fixed-layout Nano UDP packets.
tests = functiontests(localfunctions);
end

function testCommandPayloadRoundTrip(~)
pulseUs = uint16([1000, 1250, 1500, 1750]);
payloadTemplate = NanoUdpBatchProtocol.buildCommandTemplate(uint32(17), uint8(15), pulseUs);
payloadBytes = NanoUdpBatchProtocol.setCommandHostTxUs(payloadTemplate, uint32(123456789));
decodedPayload = NanoUdpBatchProtocol.decodeCommandPayload(payloadBytes);

verifyEqualPayload(decodedPayload.sampleSequence, uint32(17));
verifyEqualPayload(decodedPayload.hostTxUs, uint32(123456789));
verifyEqualPayload(decodedPayload.activeMask, uint8(15));
verifyEqualPayload(decodedPayload.pulseUs, pulseUs);
end

function testTelemetryPayloadRoundTrip(~)
payloadBytes = NanoUdpBatchProtocol.buildTelemetryPayload( ...
    uint32(29), ...
    uint32(10101), ...
    uint32(20202), ...
    uint32(30303), ...
    uint32(40404), ...
    uint8(5), ...
    uint8(2));
decodedPayload = NanoUdpBatchProtocol.decodeTelemetryPayload(payloadBytes);

verifyEqualPayload(decodedPayload.sampleSequence, uint32(29));
verifyEqualPayload(decodedPayload.hostTxUs, uint32(10101));
verifyEqualPayload(decodedPayload.boardRxUs, uint32(20202));
verifyEqualPayload(decodedPayload.applyStartUs, uint32(30303));
verifyEqualPayload(decodedPayload.applyEndUs, uint32(40404));
verifyEqualPayload(decodedPayload.activeMask, uint8(5));
verifyEqualPayload(decodedPayload.statusCode, uint8(2));
end

function testTelemetryTextLogFormatting(~)
payloadBytes = NanoUdpBatchProtocol.buildTelemetryPayload( ...
    uint32(3), ...
    uint32(4), ...
    uint32(5), ...
    uint32(6), ...
    uint32(7), ...
    uint8(15), ...
    uint8(0));
payloadLine = NanoUdpBatchProtocol.formatPayloadForTextLog(payloadBytes);

verifyEqualPayload(payloadLine, "BATCH_EVENT,3,4,5,6,7,15,0");
end

function verifyEqualPayload(actualValue, expectedValue)
assert(isequal(actualValue, expectedValue), "NanoUdpBatchProtocol test assertion failed.");
end
