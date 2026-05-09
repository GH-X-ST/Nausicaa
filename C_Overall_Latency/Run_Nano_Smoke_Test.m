%RUN_NANO_SMOKE_TEST Quick serial check for the Nano transmitter.
serialPort = "COM11";
baudRate = 1000000;

s = serialport(serialPort, baudRate);
cleanupObj = onCleanup(@() clear("s"));
configureTerminator(s, "LF");
flush(s);

writeline(s, "HELLO");
disp(readline(s));

writeline(s, "SET_NEUTRAL");
disp(readline(s));

writeline(s, "STATUS");
disp(readline(s));

