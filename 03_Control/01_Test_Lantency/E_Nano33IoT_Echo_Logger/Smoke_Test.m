client = tcpclient("192.168.0.33", 9500, "ConnectTimeout", 5, "Timeout", 5);
pause(0.5);
disp(strtrim(readline(client)));
clear client
