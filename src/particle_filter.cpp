/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <limits>
#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <map>

#include "particle_filter.h"
using namespace std;
void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 50;
    default_random_engine gen;
    normal_distribution<double> N_x_init(x, std[0]);
    normal_distribution<double> N_y_init(y, std[1]);
    normal_distribution<double> N_theta_init(theta, std[2]);
    for (int i=0; i<num_particles; i++) {
        Particle p;
        p.id = i;
        p.x = N_x_init(gen);
        p.y = N_y_init(gen);
        p.theta = N_theta_init(gen);
        p.weight = 1.0;
        particles.push_back(p);
        weights.push_back(1.0);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    default_random_engine gen;
    double x, y, theta;

    for (int i=0; i<num_particles; i++) {
        Particle &p = particles.at(i);
        
        // handle divide by zero case
        if (fabs(yaw_rate) < 0.00001) {
            x = p.x + velocity*delta_t*cos(p.theta);
            y =  p.y + velocity*delta_t*sin(p.theta);
            theta = p.theta;            
        } else {
            x = p.x + (velocity/yaw_rate)*(sin(p.theta + yaw_rate*delta_t) - sin(p.theta));
            y = p.y + (velocity/yaw_rate)*(cos(p.theta) - cos(p.theta + yaw_rate*delta_t));
            theta = p.theta + yaw_rate*delta_t;
        }

        // add noise from Gaussian distribution
        normal_distribution<double> N_x_init(x, std_pos[0]);
        normal_distribution<double> N_y_init(y, std_pos[1]);
        normal_distribution<double> N_theta_init(theta, std_pos[2]);
        p.x = N_x_init(gen);
        p.y = N_y_init(gen); 
        p.theta = N_theta_init(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    
    // Nearest neighbour search.
    // For each observation, go through all the predicted points, and find out min distance.
    for (int i=0; i<observations.size(); i++) {
        LandmarkObs& observation = observations.at(i);
        double  min_distance = numeric_limits<double>::max();
        for (const LandmarkObs& p:predicted) {
            double distance = dist(p.x, p.y, observation.x, observation.y);
            if (distance < min_distance){
                min_distance = distance;
                observation.id = p.id;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
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

    vector<LandmarkObs> transformedObjects;
    transformedObjects.resize(observations.size());

    for (int i=0; i<num_particles; ++i){
        Particle &particle = particles[i]; // reference of a particle
        // Step1: Transform the observations to Map coordinates
        for (int j=0; j < observations.size(); ++j) {
            LandmarkObs transformedObject;
            transformedObject.x = observations[j].x * cos(particle.theta) - observations[j].y * sin(particle.theta) + particle.x;
            transformedObject.y = observations[j].x * sin(particle.theta) + observations[j].y * cos(particle.theta) + particle.y;
            transformedObjects[j] = transformedObject;
        }

        std::vector<LandmarkObs> landMarks;
        std::map<int, Map::single_landmark_s> landMarkIndexMap;

        for (const auto& landmark:map_landmarks.landmark_list){
            double distance = dist(landmark.x_f, landmark.y_f, particle.x, particle.y);
            if (distance <= sensor_range){
                landMarks.push_back(LandmarkObs{ landmark.id_i,landmark.x_f,landmark.y_f });
                landMarkIndexMap.insert(std::make_pair(landmark.id_i, landmark)); // log the landmark so that it can be used later
            }
        }

        if (landMarks.size() > 0) {
            // Step 2: Do data associations
            dataAssociation(landMarks, transformedObjects);
            particle.weight = 1.0;
            for (int k=0; k<transformedObjects.size(); k++) {
                LandmarkObs observation = transformedObjects.at(k);
                double mu_x = landMarkIndexMap[observation.id].x_f;
                double mu_y = landMarkIndexMap[observation.id].y_f;
                double x = observation.x;
                double y = observation.y;
                double x_landmark = std_landmark[0];
                double y_landmark = std_landmark[1];
                double x_diff = (x - mu_x) * (x - mu_x) / (2 * x_landmark * x_landmark);
                double y_diff = (y - mu_y) * (y - mu_y) / (2 * y_landmark * y_landmark);
                particle.weight *= 1 / (2 * M_PI * x_landmark * y_landmark) * exp(-(x_diff + y_diff));
            }
            weights[i] = particle.weight;
        } else{
            weights[i] = 0.0;
        }
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    std::random_device randomDevice;
    std::mt19937 gen(randomDevice());
    std::discrete_distribution<> d(weights.begin(), weights.end());
    vector<Particle> particlesTemp;

    particlesTemp.resize(num_particles);
    for (int i=0; i<num_particles; i++) {
        particlesTemp[i] = particles[d(gen)];
    }
    particles = particlesTemp;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
