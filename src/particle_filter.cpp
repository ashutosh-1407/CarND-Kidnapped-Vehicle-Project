/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  
  num_particles = 200;  // TODO: Set the number of particles
  weights.resize(num_particles, 1.0);

  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  for(int i=0; i<num_particles; ++i) {
    struct Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    particles.push_back(p);
  }
  
  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  
  std::default_random_engine gen;
  
  for(int i=0; i<num_particles; ++i) {
    double x_0 = particles[i].x;
    double y_0 = particles[i].y;
    double theta_0 = particles[i].theta;
    double x_f, y_f, theta_f;
    if (fabs(yaw_rate) > 0.0001) {
      x_f = x_0 + (velocity / yaw_rate) * (sin(theta_0 + yaw_rate * delta_t) - sin(theta_0));
      y_f = y_0 + (velocity / yaw_rate) * (cos(theta_0) - cos(theta_0 + yaw_rate * delta_t));
      theta_f = theta_0 + yaw_rate * delta_t;
    }
    else {
      x_f = x_0 + velocity * cos(theta_0) * delta_t;
      y_f = y_0 + velocity * sin(theta_0) * delta_t;
      theta_f = theta_0 + yaw_rate * delta_t;
    }
    std::normal_distribution<double> dist_x(x_f, std_pos[0]);
    std::normal_distribution<double> dist_y(y_f, std_pos[1]);
    std::normal_distribution<double> dist_theta(theta_f, std_pos[2]);
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  
  for(int i=0; i<num_particles; ++i) {
    double x_part = particles[i].x;
    double y_part = particles[i].y;
    double theta_part = particles[i].theta;
    double weight_part = 1;
    std::vector<int> assoc;
    std::vector<double> x_maps;
    std::vector<double> y_maps;
    for(unsigned int j=0; j<observations.size(); ++j) {
      double obs_x = observations[j].x;
      double obs_y = observations[j].y;
      double x_map = x_part + (cos(theta_part) * obs_x) - (sin(theta_part) * obs_y);
      double y_map = y_part + (sin(theta_part) * obs_x) + (cos(theta_part) * obs_y);
      double min_dist = std::numeric_limits<const float>::infinity();
      int landmark_id;
      double landmark_x;
      double landmark_y;
      for(unsigned int k=0; k<map_landmarks.landmark_list.size(); ++k) {
        double curr_dist = dist(x_map, y_map, map_landmarks.landmark_list[k].x_f, map_landmarks.landmark_list[k].y_f);
        if (curr_dist < min_dist and curr_dist <= sensor_range) {
          min_dist = curr_dist;
          landmark_id = map_landmarks.landmark_list[k].id_i;
          landmark_x = map_landmarks.landmark_list[k].x_f;
          landmark_y = map_landmarks.landmark_list[k].y_f;
        }
      }
      assoc.push_back(landmark_id);
      x_maps.push_back(x_map);
      y_maps.push_back(y_map);
      double gauss_norm;
      gauss_norm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
      double exponent;
      exponent = (pow(x_map - landmark_x, 2) / (2 * pow(std_landmark[0], 2)))
                  + (pow(y_map - landmark_y, 2) / (2 * pow(std_landmark[1], 2)));
      double wt;
      wt = gauss_norm * exp(-exponent);
      weight_part*=wt;
    }
    SetAssociations(particles[i], assoc, x_maps, y_maps);
    particles[i].weight = weight_part;
    weights[i] = weight_part;
  }

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  std::vector<Particle> parts;
  std::default_random_engine rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> d(weights.begin(), weights.end());

  for(int i=0; i<num_particles; ++i) {
    int random_number = d(gen);
    parts.push_back(particles[random_number]);
  }

  particles = parts;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}