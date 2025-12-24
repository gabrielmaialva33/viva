/**
 * WordRotate Hook
 * Cycles through words with slide animation
 * Inspired by Metronic's rotating headline
 */
export const WordRotate = {
  mounted() {
    this.words = JSON.parse(this.el.dataset.words || '[]');
    this.duration = parseInt(this.el.dataset.duration) || 2500;
    this.currentIndex = 0;
    this.wordElement = this.el.querySelector('[data-word-display]');

    if (this.words.length > 0 && this.wordElement) {
      this.wordElement.textContent = this.words[0];
      this.startRotation();
    }
  },

  startRotation() {
    this.interval = setInterval(() => {
      this.rotateWord();
    }, this.duration);
  },

  rotateWord() {
    // Add exit animation
    this.wordElement.classList.add('word-exit');
    this.wordElement.classList.remove('word-enter');

    setTimeout(() => {
      // Change word
      this.currentIndex = (this.currentIndex + 1) % this.words.length;
      this.wordElement.textContent = this.words[this.currentIndex];

      // Add enter animation
      this.wordElement.classList.remove('word-exit');
      this.wordElement.classList.add('word-enter');
    }, 250); // Half of animation duration
  },

  destroyed() {
    if (this.interval) {
      clearInterval(this.interval);
    }
  }
};
