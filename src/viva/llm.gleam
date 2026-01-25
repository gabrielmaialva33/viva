import gleam/result

@external(erlang, "Elixir.Viva.Llm", "add")
pub fn add(a: Int, b: Int) -> Int
