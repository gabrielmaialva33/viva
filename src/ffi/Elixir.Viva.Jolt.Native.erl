%% VIVA JoltPhysics NIF Wrapper
%% AAA Physics Engine for Gleam - Complete API
-module('Elixir.Viva.Jolt.Native').
-export([
    %% World
    world_new/0,
    world_step/2,
    world_step_n/3,
    world_optimize/1,
    body_count/1,
    tick/1,

    %% Body creation
    create_box/4,
    create_sphere/4,
    create_capsule/5,
    create_cylinder/5,

    %% Position & Rotation
    get_position/2,
    set_position/3,
    get_rotation/2,
    set_rotation/3,

    %% Velocity
    get_velocity/2,
    set_velocity/3,
    get_angular_velocity/2,
    set_angular_velocity/3,

    %% Forces & Impulses
    add_force/3,
    add_torque/3,
    add_impulse/3,
    add_angular_impulse/3,

    %% Kinematic movement
    move_kinematic/5,

    %% Body properties
    is_active/2,
    activate_body/2,
    deactivate_body/2,
    set_friction/3,
    get_friction/2,
    set_restitution/3,
    get_restitution/2,
    set_gravity_factor/3,
    get_gravity_factor/2,

    %% Raycast
    cast_ray/3,

    %% Contact Events
    get_contacts/1,
    has_contacts/1,

    %% Utility
    native_check/0
]).
-on_load(init/0).

-define(NIF_PATH, "native/viva_jolt/target/release/libviva_jolt").

init() ->
    PrivDir = code:priv_dir(viva),
    NifPath = case PrivDir of
        {error, _} ->
            ?NIF_PATH;
        Dir ->
            filename:join(Dir, "native/libviva_jolt")
    end,
    erlang:load_nif(NifPath, 0).

%% World
world_new() -> erlang:nif_error(nif_not_loaded).
world_step(_World, _Dt) -> erlang:nif_error(nif_not_loaded).
world_step_n(_World, _N, _Dt) -> erlang:nif_error(nif_not_loaded).
world_optimize(_World) -> erlang:nif_error(nif_not_loaded).
body_count(_World) -> erlang:nif_error(nif_not_loaded).
tick(_World) -> erlang:nif_error(nif_not_loaded).

%% Body creation
create_box(_World, _Pos, _HalfExtents, _MotionType) -> erlang:nif_error(nif_not_loaded).
create_sphere(_World, _Pos, _Radius, _MotionType) -> erlang:nif_error(nif_not_loaded).
create_capsule(_World, _Pos, _HalfHeight, _Radius, _MotionType) -> erlang:nif_error(nif_not_loaded).
create_cylinder(_World, _Pos, _HalfHeight, _Radius, _MotionType) -> erlang:nif_error(nif_not_loaded).

%% Position & Rotation
get_position(_World, _Index) -> erlang:nif_error(nif_not_loaded).
set_position(_World, _Index, _Pos) -> erlang:nif_error(nif_not_loaded).
get_rotation(_World, _Index) -> erlang:nif_error(nif_not_loaded).
set_rotation(_World, _Index, _Rot) -> erlang:nif_error(nif_not_loaded).

%% Velocity
get_velocity(_World, _Index) -> erlang:nif_error(nif_not_loaded).
set_velocity(_World, _Index, _Vel) -> erlang:nif_error(nif_not_loaded).
get_angular_velocity(_World, _Index) -> erlang:nif_error(nif_not_loaded).
set_angular_velocity(_World, _Index, _Vel) -> erlang:nif_error(nif_not_loaded).

%% Forces & Impulses
add_force(_World, _Index, _Force) -> erlang:nif_error(nif_not_loaded).
add_torque(_World, _Index, _Torque) -> erlang:nif_error(nif_not_loaded).
add_impulse(_World, _Index, _Impulse) -> erlang:nif_error(nif_not_loaded).
add_angular_impulse(_World, _Index, _Impulse) -> erlang:nif_error(nif_not_loaded).

%% Kinematic movement
move_kinematic(_World, _Index, _TargetPos, _TargetRot, _Dt) -> erlang:nif_error(nif_not_loaded).

%% Body properties
is_active(_World, _Index) -> erlang:nif_error(nif_not_loaded).
activate_body(_World, _Index) -> erlang:nif_error(nif_not_loaded).
deactivate_body(_World, _Index) -> erlang:nif_error(nif_not_loaded).
set_friction(_World, _Index, _Friction) -> erlang:nif_error(nif_not_loaded).
get_friction(_World, _Index) -> erlang:nif_error(nif_not_loaded).
set_restitution(_World, _Index, _Restitution) -> erlang:nif_error(nif_not_loaded).
get_restitution(_World, _Index) -> erlang:nif_error(nif_not_loaded).
set_gravity_factor(_World, _Index, _Factor) -> erlang:nif_error(nif_not_loaded).
get_gravity_factor(_World, _Index) -> erlang:nif_error(nif_not_loaded).

%% Raycast
cast_ray(_World, _Origin, _Direction) -> erlang:nif_error(nif_not_loaded).

%% Contact Events
get_contacts(_World) -> erlang:nif_error(nif_not_loaded).
has_contacts(_World) -> erlang:nif_error(nif_not_loaded).

%% Utility
native_check() -> erlang:nif_error(nif_not_loaded).
