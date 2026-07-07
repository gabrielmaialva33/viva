import viva/memory/hrr

pub fn bind_unbind_roundtrip_test() {
  let a = hrr.random(512)
  let b = hrr.random(512)
  let assert Ok(trace) = hrr.bind(a, b)
  let assert Ok(recovered) = hrr.unbind(trace, a)
  assert hrr.similarity(recovered, b) >. 0.5
}

pub fn similarity_self_test() {
  let a = hrr.random(128)
  assert hrr.similarity(a, a) >. 0.99
}

pub fn from_list_roundtrip_test() {
  let h = hrr.from_list([1.0, 0.0, 0.0, 0.0])
  assert hrr.dim(h) == 4
  assert hrr.to_list(h) == [1.0, 0.0, 0.0, 0.0]
}
