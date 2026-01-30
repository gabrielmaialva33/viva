%% VIVA Senses FFI - Shell command execution
-module(viva_senses_ffi).
-export([run_shell/1]).

%% Run shell command and return output as binary
-spec run_shell(binary()) -> binary().
run_shell(CmdBin) ->
    Cmd = binary_to_list(CmdBin),
    Result = os:cmd(Cmd),
    list_to_binary(Result).
