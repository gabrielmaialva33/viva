import viva/neural/activation.{Sigmoid}
import viva/neural/network
import viva_tensor/tensor

pub fn network_forward_shape_test() {
  let assert Ok(net) = network.new([2, 3, 1], Sigmoid, Sigmoid)
  let input = tensor.from_list([0.5, -0.25])
  let assert Ok(out) = network.predict(net, input)
  assert tensor.shape(out) == [1]
}

pub fn param_count_test() {
  let assert Ok(net) = network.new([2, 3, 1], Sigmoid, Sigmoid)
  // (2*3 + 3) + (3*1 + 1)
  assert network.param_count(net) == 13
}
