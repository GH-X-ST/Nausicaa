ipAddress = "192.168.0.33";
port = 9500;

socket = javaObject("java.net.DatagramSocket");
cleanupSocket = onCleanup(@() socket.close());
socket.setSoTimeout(int32(3000));

remoteAddress = javaMethod("getByName", "java.net.InetAddress", char(ipAddress));
payloadBytes = uint8("HELLO");
outboundPacket = javaObject( ...
    "java.net.DatagramPacket", ...
    int8(payloadBytes), ...
    numel(payloadBytes), ...
    remoteAddress, ...
    int32(port));
socket.send(outboundPacket);

receivePacket = javaObject("java.net.DatagramPacket", int8(zeros(1, 512)), 512);
socket.receive(receivePacket);
packetBytes = uint8(receivePacket.getData());
packetLength = double(receivePacket.getLength());
packetBytes = reshape(packetBytes(1:packetLength), 1, []);
responseText = string(strtrim(char(packetBytes)));

disp(responseText)
