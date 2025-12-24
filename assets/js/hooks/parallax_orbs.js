/**
 * ParallaxOrbs Hook
 * Makes floating orbs follow the mouse cursor with smooth animation
 * Inspired by Metronic's hero section
 */
export const ParallaxOrbs = {
  mounted() {
    this.orbs = this.el.querySelectorAll('[data-parallax-orb]');
    this.mouseX = 0;
    this.mouseY = 0;
    this.targetX = 0;
    this.targetY = 0;
    this.animating = false;

    // Get container dimensions
    this.updateDimensions();

    // Bind event handlers
    this.handleMouseMove = this.handleMouseMove.bind(this);
    this.handleMouseLeave = this.handleMouseLeave.bind(this);
    this.animate = this.animate.bind(this);

    // Add listeners
    this.el.addEventListener('mousemove', this.handleMouseMove);
    this.el.addEventListener('mouseleave', this.handleMouseLeave);
    window.addEventListener('resize', () => this.updateDimensions());

    // Start animation loop
    this.startAnimation();
  },

  updateDimensions() {
    const rect = this.el.getBoundingClientRect();
    this.centerX = rect.width / 2;
    this.centerY = rect.height / 2;
  },

  handleMouseMove(e) {
    const rect = this.el.getBoundingClientRect();
    // Calculate mouse position relative to center (-1 to 1)
    this.targetX = (e.clientX - rect.left - this.centerX) / this.centerX;
    this.targetY = (e.clientY - rect.top - this.centerY) / this.centerY;
  },

  handleMouseLeave() {
    // Smoothly return to center
    this.targetX = 0;
    this.targetY = 0;
  },

  startAnimation() {
    if (!this.animating) {
      this.animating = true;
      this.animate();
    }
  },

  animate() {
    if (!this.animating) return;

    // Smooth interpolation (easing)
    const ease = 0.08;
    this.mouseX += (this.targetX - this.mouseX) * ease;
    this.mouseY += (this.targetY - this.mouseY) * ease;

    // Apply transforms to each orb
    this.orbs.forEach(orb => {
      const speed = parseFloat(orb.dataset.parallaxSpeed) || 50;
      const invert = orb.dataset.parallaxInvert === 'true' ? -1 : 1;

      const moveX = this.mouseX * speed * invert;
      const moveY = this.mouseY * speed * invert;

      orb.style.transform = `translate(${moveX}px, ${moveY}px)`;
    });

    requestAnimationFrame(this.animate);
  },

  destroyed() {
    this.animating = false;
    this.el.removeEventListener('mousemove', this.handleMouseMove);
    this.el.removeEventListener('mouseleave', this.handleMouseLeave);
  }
};
