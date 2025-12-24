/**
 * StepProgress Hook
 * Auto-advancing steps with progress bar animation
 * Inspired by Metronic's "How It Works" section
 */
export const StepProgress = {
  mounted() {
    this.steps = this.el.querySelectorAll('[data-step]');
    this.progressBars = this.el.querySelectorAll('[data-progress-bar]');
    this.contents = this.el.querySelectorAll('[data-step-content]');
    this.stepDuration = parseInt(this.el.dataset.stepDuration) || 5000;
    this.currentStep = 0;
    this.isPaused = false;
    this.progress = 0;

    // Bind handlers
    this.handleStepClick = this.handleStepClick.bind(this);

    // Add click listeners to steps
    this.steps.forEach((step, index) => {
      step.addEventListener('click', () => this.handleStepClick(index));
    });

    // Initialize first step
    this.activateStep(0);
    this.startAutoAdvance();
  },

  handleStepClick(index) {
    this.isPaused = true;
    this.activateStep(index);

    // Resume auto-advance after 4 seconds
    setTimeout(() => {
      this.isPaused = false;
      this.startAutoAdvance();
    }, 4000);
  },

  activateStep(index) {
    this.currentStep = index;
    this.progress = 0;

    // Update step styles
    this.steps.forEach((step, i) => {
      if (i === index) {
        step.classList.add('step-active');
        step.classList.remove('step-inactive');
      } else {
        step.classList.remove('step-active');
        step.classList.add('step-inactive');
      }
    });

    // Update content visibility
    this.contents.forEach((content, i) => {
      if (i === index) {
        content.classList.remove('hidden');
        content.classList.add('step-content-enter');
      } else {
        content.classList.add('hidden');
        content.classList.remove('step-content-enter');
      }
    });

    // Reset and animate progress bar
    this.progressBars.forEach((bar, i) => {
      if (i === index) {
        bar.style.width = '0%';
      } else {
        bar.style.width = i < index ? '100%' : '0%';
      }
    });
  },

  startAutoAdvance() {
    // Clear existing interval
    if (this.interval) clearInterval(this.interval);
    if (this.progressInterval) clearInterval(this.progressInterval);

    // Progress bar animation
    this.progressInterval = setInterval(() => {
      if (this.isPaused) return;

      this.progress += (100 / (this.stepDuration / 50));

      if (this.progressBars[this.currentStep]) {
        this.progressBars[this.currentStep].style.width = `${Math.min(this.progress, 100)}%`;
      }
    }, 50);

    // Step advancement
    this.interval = setInterval(() => {
      if (this.isPaused) return;

      const nextStep = (this.currentStep + 1) % this.steps.length;
      this.activateStep(nextStep);
    }, this.stepDuration);
  },

  destroyed() {
    if (this.interval) clearInterval(this.interval);
    if (this.progressInterval) clearInterval(this.progressInterval);
  }
};
