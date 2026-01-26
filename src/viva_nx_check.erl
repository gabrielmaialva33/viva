%% VIVA Nx Check - Check if Nx acceleration is available
%% Pure Gleam implementation - Nx not available

-module(viva_nx_check).
-export([available/0]).

%% Check if Nx module is available
%% Returns false for Pure Gleam (no Elixir/Nx dependency)
available() ->
    false.
