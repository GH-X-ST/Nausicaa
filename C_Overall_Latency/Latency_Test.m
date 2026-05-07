function runData = Latency_Test(config)
%LATENCY_TEST Select a command profile and run the reusable command path.
arguments
    config (1,1) struct = struct()
end

config = normalizeEntryConfig(config);

switch config.mode
    case "deflection"
        commandFcn = @Test_Deflection_Profile;
    case "latency"
        commandFcn = @Test_Latency_Profile;
    case "external"
        if ~isfield(config, "commandFcn") || ~isa(config.commandFcn, "function_handle")
            error("Latency_Test:MissingExternalCommandFcn", ...
                "mode='external' requires config.commandFcn with the command-provider signature.");
        end
        commandFcn = config.commandFcn;
    otherwise
        error("Latency_Test:InvalidMode", ...
            "config.mode must be 'deflection', 'latency', or 'external'.");
end

runData = Run_Control_Path(config, commandFcn);
end

function config = normalizeEntryConfig(config)
if ~isfield(config, "mode") || strlength(string(config.mode)) == 0
    config.mode = "latency";
else
    config.mode = lower(string(config.mode));
end
end
