import viva/soul/soul
import viva_emotion/stimulus

pub fn soul_lifecycle_test() {
  let assert Ok(s) = soul.start(1)
  assert soul.is_alive(s)

  let pad = soul.get_pad(s)
  assert pad.pleasure >=. -1.0 && pad.pleasure <=. 1.0

  soul.feel(s, stimulus.Acceptance, 0.5)
  soul.tick(s, 0.1)

  let pad2 = soul.get_pad(s)
  assert pad2.arousal >=. -1.0 && pad2.arousal <=. 1.0

  soul.kill(s)
}
