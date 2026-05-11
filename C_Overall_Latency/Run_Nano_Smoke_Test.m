%RUN_NANO_SMOKE_TEST Quick serial check for the Nano transmitter.
%% =========================================================================
% SECTION MAP
% ==========================================================================
% 1) Serial-port configuration
% 2) ASCII command probe
% ==========================================================================
%% =========================================================================
% 1) Serial-port configuration
% ==========================================================================
% The Nano firmware uses LF-terminated ASCII only for diagnostics; high-rate commands use binary packets.
serialPort = "COM11";
baudRate = 1000000;

s = serialport(serialPort, baudRate);
cleanupObj = onCleanup(@() clear("s"));
configureTerminator(s, "LF");
flush(s);

%% =========================================================================
% 2) ASCII command probe
% ==========================================================================
writeline(s, "HELLO");
disp(readline(s));

writeline(s, "SET_NEUTRAL");
disp(readline(s));

writeline(s, "STATUS");
disp(readline(s));
