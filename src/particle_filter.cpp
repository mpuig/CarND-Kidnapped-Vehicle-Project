/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <cfloat>
#include <math.h>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// Set number of particles
	num_particles = 30;

	// Initialise particles
	default_random_engine generator;
	normal_distribution<double> x_dist(x, std[0]);
	normal_distribution<double> y_dist(y, std[1]);
	normal_distribution<double> theta_dist(theta, std[2]);

	// initialize each particle
	for (int i = 0; i < num_particles; i++) {
		Particle particle;
		particle.id = i;
		particle.x = x_dist(generator);
		particle.y = y_dist(generator);
		particle.theta = theta_dist(generator);
		particle.weight = 1.0;

		particles.push_back(particle);
		weights.push_back(particle.weight);
	}
	// particle filter initialized
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine generator;
	normal_distribution<double> x_noise(0, std_pos[0]);
	normal_distribution<double> y_noise(0, std_pos[1]);
	normal_distribution<double> theta_noise(0, std_pos[2]);

	// update each particle
	for (auto& particle : particles) {
		double new_x = particle.x;
		double new_y = particle.y;
		// avoid division by zero
		if (fabs(yaw_rate) < 1e-5) {
			new_x += delta_t * velocity * cos(particle.theta);
			new_y += delta_t * velocity * sin(particle.theta);
		} else {
			new_x += velocity / yaw_rate * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
			new_y += velocity / yaw_rate * (-cos(particle.theta + yaw_rate * delta_t) + cos(particle.theta));
		}
		particle.x = new_x + x_noise(generator);
		particle.y = new_y + y_noise(generator);
		particle.theta += delta_t * yaw_rate + theta_noise(generator);
	}
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	// Loop over each observation
	for (auto& observation : observations) {
		double smallest_distance = numeric_limits<double>::max();
		// Loop over each predicted measurement
		for (int i=0; i<predicted.size(); ++i) {
			// Calculate squared distance
			const double dist = pow(predicted[i].x - observation.x, 2) +
													pow(predicted[i].y - observation.y, 2);
			if (dist < smallest_distance) {
				smallest_distance = dist;
				observation.id = i;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

	if (observations.size() < 1) {
		return;
	}

	const double squared_range = sensor_range * sensor_range;
	const double squared_sigma = std_landmark[0] * std_landmark[0];
	const double norm = 1.0 / (2.0 * M_PI * squared_sigma);
	const double coeff1 = 1.0 / (2 * std_landmark[0] * std_landmark[0]);
	const double coeff2 = 1.0 / (2 * std_landmark[1] * std_landmark[1]);

	// Update each particle
	LandmarkObs obs;
	for (int i=0; i<num_particles; ++i) {

		// For readibility
		Particle particle = particles[i];

		// Used when convert to map coordinates
    double s = sin(particle.theta);
    double c = cos(particle.theta);

		// Build vector of predicted observations (those in sensor_range)
		// candidate map landmarks to be spotted
		vector<LandmarkObs> predicted;
		for (auto& landmark : map_landmarks.landmark_list) {
			if (dist(landmark.x_f, particle.x, landmark.y_f, particle.x) <= squared_range) {
				obs.x = landmark.x_f;
				obs.y = landmark.y_f;
				obs.id = landmark.id_i;
				predicted.push_back(obs);
			}
		}

		// Build vector of observations from car coordinates to map coordinates
		vector<LandmarkObs> obs_in_map_coords;
		for (auto& observation : observations) {
			obs.x = observation.x * c - observation.y * s + particle.x;
			obs.y = observation.x * s + observation.y * c + particle.y;
			obs.id = observation.id;
			obs_in_map_coords.push_back(obs);
		}

		// Don't update the weight if there are no predicted observations
		if (predicted.size() < 1) {
			continue;
		}
		// Assign each observation to nearest landmark on map
		dataAssociation(predicted, obs_in_map_coords);

		// Calculate particles weights
		double w = 1.0;
		for (auto& o : obs_in_map_coords) {
		    w *= norm * exp (-(coeff1 * pow(o.x - predicted[o.id].x, 2) + coeff2 * pow(o.y - predicted[o.id].y, 2)));
		}

		// Update particle weight
		particles[i].weight = w;
		weights[i] = w;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	vector<Particle> new_particles;
	default_random_engine gen;
	discrete_distribution<> dist(weights.begin(), weights.end());

	// Resample particles
	for (int i=0; i<num_particles; ++i) {
		new_particles.push_back(particles[dist(gen)]);
	}
	particles = new_particles;
}

void ParticleFilter::write(string filename) {
	// You don't need to modify this file.
	ofstream dataFile;
	dataFile.open(filename, ios::app);
	for (int i=0; i<num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
