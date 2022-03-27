// The Nature of Code
// Daniel Shiffman
// http://natureofcode.com

// A class to describe a population of particles

class Population {
  constructor(total) {
    // Number of generations
    this.generations = 0;
    // Empty array with length total
    this.population = new Array(total);
    // Fill array with particles
    this.population = this.population.fill(new Particle())
  }

  update() {
    this.population.forEach(p => {
      p.think();
      p.update();
    });
  }

  show() {
    this.population.forEach(p => {
      p.show();
    });
  }

  reproduce() {
    const brainA = this.pickOne();
    const brainB = this.pickOne();
    const childBrain = brainA.crossover(brainB);
    // 1% mutation rate
    childBrain.mutate(0.01);
    return new Particle(childBrain);
  }

  // Pick one parent probability according to normalized fitness
  pickOne() {
    let index = 0;
    let r = random(1);
    while (r > 0) {
      r -= this.population[index].fitness;
      index += 1;
    }
    index -= 1;
    return this.population[index].brain;
  }

  // Normalize all fitness values
  calculateFitness() {
    const sum = this.population.reduce((total, p) => total + p.calcFitness(), 0);
    this.population = this.population.map(p => ({
      ...p,
      fitness: p.fitness / sum
    }));
  }

  // Making the next generation
  reproduction() {
    // Refill the population with children from the mating pool
    this.population = this.population.map( () => this.reproduce() );
    this.generations += 1;
  }
}
