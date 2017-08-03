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
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
    num_particles = 50;
    default_random_engine gen; 
    normal_distribution<double> dist_x(x, std[0]); 
    normal_distribution<double> dist_y(y, std[1]); 
    normal_distribution<double> dist_theta(theta, std[2]); 
    
    for (int i = 0; i < num_particles; ++i){
        Particle p;
        p.id = i;
        p.x = dist_x(gen);                
        p.y = dist_y(gen);                
        p.theta = dist_theta(gen);
        cout << p.x << endl; 
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

    for (int i = 0; i < num_particles; ++i){
        Particle p = particles[i];
        double theta_f = p.theta + delta_t*yaw_rate;
        double delta_x, delta_y;
        if (yaw_rate < 0.001) {
            delta_x = velocity*cos(p.theta)*delta_t; 
            delta_y = velocity*sin(p.theta)*delta_t; 
        }
        else {
            double v = velocity/yaw_rate;
            delta_x = v*(sin(theta_f) - sin(p.theta));
            delta_y = v*(cos(p.theta) - cos(theta_f));
        }

        double x_f = p.x + delta_x; 
        double y_f = p.y + delta_y;
        normal_distribution<double> dist_x(x_f, std_pos[0]); 
        normal_distribution<double> dist_y(y_f, std_pos[1]); 
        normal_distribution<double> dist_theta(theta_f, std_pos[2]); 
        particles[i].x = dist_x(gen);                
        particles[i].y = dist_y(gen);                
        particles[i].theta = dist_theta(gen); 
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs>& observations, std::vector<LandmarkObs> map_landmarks) {
	// TODO: Find the landmark that is closest to each observed measurement and assign the 
	//   that particular landmark to the observed measurement.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    for (int i = 0; i < observations.size(); ++i){
        LandmarkObs obs = observations[i]; // measured
        LandmarkObs gt = map_landmarks[0]; // ground truth
        double nearest_dist = dist(gt.x, gt.y, obs.x, obs.y);
        observations[i].id = 0;
        for (int j = 1; j < map_landmarks.size(); ++j){
            gt = map_landmarks[j];
            double d = dist(gt.x, gt.y, obs.x, obs.y);
            if (d < nearest_dist){
                nearest_dist = d;
                observations[i].id = j;
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
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    for (int i = 0; i < num_particles; ++i){
        Particle p = particles[i];

        // transform observations to map coordinates
        std::vector<LandmarkObs> observations_tf;
        for (int j = 0; j < observations.size(); ++j){
            LandmarkObs obs = observations[j];
            LandmarkObs l;
            l.x = p.x + obs.x * cos(p.theta) - obs.y * sin(p.theta);
            l.y = p.y + obs.x * sin(p.theta) + obs.y * cos(p.theta);
            observations_tf.push_back(l);
        }

        // find landmarks within sensor range of each particle
        std::vector<LandmarkObs> landmarks;
        int n_landmarks = map_landmarks.landmark_list.size();
        int id = 0;
        for (int j =0; j < n_landmarks; ++j){
            Map::single_landmark_s l = map_landmarks.landmark_list[j];
            double dist_from_particle = dist(p.x, p.y, l.x_f, l.y_f); 
            if (dist_from_particle <= sensor_range) {
                LandmarkObs l_map;
                l_map.id = id++;
                l_map.x = l.x_f;
                l_map.y = l.y_f;
                landmarks.push_back(l_map);
            }
        }

        // assign landmarks to all observations
        if (landmarks.size() > 0) 
        dataAssociation(observations_tf, landmarks);

        // calculate particle's weight        
        double w;
        if (landmarks.size() == 0){
            w = 0.;
        }
        else{
            w = 1.;
            for (int j = 0; j < observations_tf.size(); ++j){
                LandmarkObs obs = observations_tf[j];
                LandmarkObs gt = landmarks[obs.id];
                // define variables for better readability 
                double sigma_x = std_landmark[0];
                double sigma_y = std_landmark[1];
                double prob = 0.5/M_PI/ sigma_x / sigma_y;
                double e_x = -0.5*pow((obs.x - gt.x) / sigma_x,2);
                double e_y = -0.5*pow((obs.y - gt.y) / sigma_y,2);
                prob *= exp(e_x + e_y);
                w *= prob;
            }
        }
        particles[i].weight = w;
        weights[i] = w;
    }    
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    default_random_engine gen;
    discrete_distribution<int> distr(weights.begin(), weights.end()); 
    std::vector<Particle> resampled_particles;
    for (int i = 0; i < num_particles; ++i){
        int index = distr(gen);
        Particle p = particles[index];
        resampled_particles.push_back(p);
    } 
    particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
