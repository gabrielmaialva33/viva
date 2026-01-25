%% viva_nx_check - Runtime detection of Nx availability
%%
%% Simple module to check if Elixir Nx is loaded.

-module(viva_nx_check).
-export([available/0]).

%% Check if Nx module is available
available() ->
    case code:ensure_loaded('Elixir.Nx') of
        {module, _} -> true;
        {error, _} -> false
    end.
