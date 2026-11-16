# Musical swarm simulator
#
# A three-dimensional Boids simulation coupled to a MIDI sonification. In every frame
# the swarm is updated, the active mapping strategy translates the swarm state into
# MIDI events, and spatial and musical descriptors are logged for later analysis.
#
# Modes:
#   BATCH_RUN = False  interactive real-time animation, data is saved when the window closes
#   BATCH_RUN = True   automated runs of MAX_FRAMES frames over a parameter sweep, as used
#                      to produce the data analysed in the thesis
#
# Note events, and thereby all musical descriptors, are only generated when a MIDI output
# port is available. The experiments used a virtual MIDI port (loopMIDI). Connecting a
# digital audio workstation to the port is only needed for listening to the output.
#
# Each run produces two files:
#   <name>_metrics.csv  one row per frame with the spatial and musical descriptors
#   <name>_rhythm.csv   one row per note onset with the inter-onset interval
#
# The experiments used the Boids behaviour and the PhaserMapping (agent-centric parameter
# mapping). PredatorPrey, SoloistMapping and HiveMindMapping are alternative strategies
# that were not used in the experiments.

import numpy as np
import mido
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d.axes3d as p3
from scipy.spatial import ConvexHull, KDTree
import time
import pandas as pd
import datetime
import os

# # # # # # # # # # # #
# 1. HELPER FUNCTIONS #
# # # # # # # # # # # #
def map_value(val, in_min, in_max, out_min, out_max):
    # Maps a value from one range to another (clamped)
    mapped = out_min + (val - in_min) * (out_max - out_min) / (in_max - in_min)
    return max(min(mapped, out_max), out_min)

def get_closest_scale_note(note, scale):
    """Snaps a raw MIDI note to a specific musical scale."""
    octave = (note // 12) * 12
    rel_note = note % 12
    closest = min(scale, key=lambda x: abs((x % 12) - rel_note))
    return int(octave + (closest % 12))

def get_swarm_volume(swarm):
    # ConvexHull needs at least dim+1 points to form a volume, otherwise return 0
    if swarm.num_agents <= swarm.dim:
        return 0.0
    else:
        try:
            hull = ConvexHull(swarm.positions)
            return hull.volume
        except:
            # In case of issues with the hull, e.g. all points are in a straight line
            return 0.0

def get_ambitus(active_notes):
    # Calculate Ambitus (Range between lowest and highest pitch). If no active notes, range is 0.
    pitch_range = 0
    if len(active_notes) > 0:
        # Extract the current active pitches from the active_notes dictionary. Handle both single notes and notes with timestamps
        current_pitches = []
        for val in active_notes.values():
            if isinstance(val, (tuple, list)):
                current_pitches.append(val[0]) # It's a tuple, take the first element
            else:
                current_pitches.append(val)
        
        pitch_range = max(current_pitches) - min(current_pitches)        
    else:
        pitch_range = 0
    return pitch_range

def get_global_polarity(swarm):
    # Polarity is a measure of how aligned the velocities are. 1 = all moving in same direction, 0 = random directions
    if swarm.num_agents == 0:
        return 0.0 

    # Calculate the Vector Sum of all velocities
    sum_vel = np.sum(swarm.velocities, axis=0) 
    magnitude_sum = np.linalg.norm(sum_vel)
    
    # Calculate the Scalar Sum of individual magnitudes
    sum_magnitudes = np.sum(np.linalg.norm(swarm.velocities, axis=1))
    
    # The Ratio
    polarity = magnitude_sum / sum_magnitudes
    
    return polarity

def get_mean_nnd(swarm):
    # Nearest Neighbor Distance (NND) is the average distance to the nearest neighbor, a measure of how tightly packed the swarm is
    if swarm.num_agents <= 1:
        return 0.0

    tree = KDTree(swarm.positions)
    distances, _ = tree.query(swarm.positions, k=2) # k=2 because the closest point will be itself with distance 0.0
    nnd = np.mean(distances[:, 1]) # Take the second column which is the nearest neighbor distance
    return nnd

def get_system_dissonance(active_notes):
    if len(active_notes) < 2: 
        return 0.0
    
    # Simplified dissonance weights per interval class, 0 to 11 semitones (Table 4.1).
    # Intervals are reduced modulo 12, so this is a pitch-class dissonance measure.
    # 0: Unison, 1: m2, 2: M2, 3: m3, 4: M3, 5: P4, 6: Tritone, 7: P5, 8: m6, 9: M6, 10: m7, 11: M7
    weights = {0: 0.0, 1: 1.0, 2: 0.6, 3: 0.2, 4: 0.1, 5: 0.1, 
               6: 0.8, 7: 0.0, 8: 0.4, 9: 0.1, 10: 0.5, 11: 0.7}
    
    total_dissonance = 0
    pairs = 0

    # Extract the current active pitches from the active_notes dictionary. Handle both single notes and notes with timestamps
    current_pitches = []
    for val in active_notes.values():
        if isinstance(val, (tuple, list)):
            current_pitches.append(val[0]) # Take the first element since it's a tuple of (note, timestamp)
        else:
            current_pitches.append(val)
    
    # Calculate dissonance for each unique pair of active notes
    for i, note_a in enumerate(current_pitches):
        for note_b in current_pitches[i+1:]:
            interval = abs(note_a - note_b) % 12
            total_dissonance += weights.get(interval, 0.5)
            pairs += 1
    if pairs > 0:
        # Average dissonance per pair of notes
        system_dissonance = total_dissonance / pairs
    else:
        system_dissonance = 0.0

    return system_dissonance

def update_simulation_data(swarm, active_notes, run_starttime=0):   
    # Logs one row of descriptors per frame. Note that run_starttime defaults to 0, so
    # relative_time is the absolute system time. Only 'timestamp' is used in the analysis.
    
    current_time = time.perf_counter()
    relative_time = current_time - run_starttime

    # Store the data points
    data_point = {
        'relative_time': relative_time,
        'swarm_volume': get_swarm_volume(swarm),
        'global_polarity': get_global_polarity(swarm),
        'mean_nnd': get_mean_nnd(swarm),
        'num_active_notes': len(active_notes),
        'system_dissonance': get_system_dissonance(active_notes),
        'ambitus': get_ambitus(active_notes),
        'separation_weight': SEPARATION_WEIGHT,
        'alignment_weight': ALIGNMENT_WEIGHT,
        'cohesion_weight': COHESION_WEIGHT, 
        'timestamp': time.perf_counter()
    }
    
    # Append the data point to the simulation data list.
    simulation_data.append(data_point)

def save_session_data(filename="run_01.csv", folder="runs" ):
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(script_dir, folder)
    os.makedirs(folder, exist_ok=True)
    path = f"{folder}/{filename}"
    # Convert metrics to CSV
    df = pd.DataFrame(simulation_data)
    df.to_csv(path + "_metrics.csv", index=False)
    
    print(f"Data stored, captured {len(df)} frames of swarm data.")
    
    # Calculate IOIs and save separately
    # IOI = time between note onset n and note onset n-1, merged across all agents
    rhythm_data = []
    for t1, t2 in zip(midi_event_timestamps, midi_event_timestamps[1:]):
        ioi_value = t2['timestamp'] - t1['timestamp']
        rhythm_data.append({
            'ioi': ioi_value,
            'polarity': t2['polarity'], # Global polarity at the second onset
            'timestamp': t2['timestamp'], # The time the interval ended
        })
    
    # Save to CSV
    ioi_df = pd.DataFrame(rhythm_data)
    ioi_df.to_csv(path + "_rhythm.csv", index=False)
        
    print(f"Rhythm data saved with {len(ioi_df)} intervals.")

def generate_filename(prefix="run"):
    run_length = len(simulation_data)
    timestamp = datetime.datetime.now().strftime("%d%m_%H%M")
    filename = f"{prefix}_N{NUM_AGENTS}_Sep{SEPARATION_WEIGHT}_Ali{ALIGNMENT_WEIGHT}_Coh{COHESION_WEIGHT}_Length{run_length}_Time{timestamp}"
    return filename

# # # # # # # # # # # #
# 2. SWARM STRATEGIES #
# # # # # # # # # # # #

# Base class for the swarm system, which holds the agents and updates them based on the chosen behavior algorithm
class Swarm:
    def __init__(self, num_agents, dim, behavior):
        self.num_agents = num_agents
        self.dim = dim
        self.behavior = behavior
        # Initialise positions and velocities randomly
        self.positions = np.random.rand(num_agents, dim) * WIDTH
        self.velocities = np.random.rand(num_agents, dim) * MAX_SPEED - (MAX_SPEED / 2)
        # Initialise the phase of each agent's internal clock
        self.phases = np.random.rand(num_agents)
    
    def set_behavior(self, new_behavior):
        #Allows swapping the algorithm on the fly
        self.behavior = new_behavior

    def update(self):
        # Delegates the update of velocities, positions and phases to the active behaviour
        self.behavior.update(self)

# Base class for swarm behaviours
class SwarmBehavior:
    #Base class for all swarm algorithms
    def update(self, swarm):
        raise NotImplementedError("Each algorithm must have an update method.")

# Boids implementation with cohesion, alignment and separation, used in the experiments.
# The three steering forces are summed (no prioritisation), added to the velocity and the
# speed is capped at MAX_SPEED. Each agent also has a phase that advances with its speed.
class Boids(SwarmBehavior):
    def __init__(self, cohesion=0.01, alignment=0.05, separation=0.4, sep_dist=10.0, vision=30.0):
        self.cohesion = cohesion
        self.alignment = alignment
        self.separation = separation
        self.sep_dist = sep_dist
        self.vision = vision

    def update(self, swarm):
        for i in range(swarm.num_agents):
            pos = swarm.positions[i]
            vel = swarm.velocities[i]
            
            # Compute distances to all other boids (naive O(N^2) neighbour search)
            diffs = swarm.positions - pos
            dists = np.linalg.norm(diffs, axis=1)
            dists[i] = np.inf # Ignore self
            
            neighbors = dists < self.vision # Vision radius
            
            if np.any(neighbors):
                # Cohesion, Alignment, Separation logic
                # Cohesion: Steer towards center of mass of neighbors
                center_of_mass = np.mean(swarm.positions[neighbors], axis=0)
                cohesion = (center_of_mass - pos) * self.cohesion
                
                # Alignment: Steer towards average velocity of neighbors
                avg_vel = np.mean(swarm.velocities[neighbors], axis=0)
                alignment = (avg_vel - vel) * self.alignment
                
                # Separation: Steer away from boids that are too close
                too_close = dists < self.sep_dist
                separation = np.zeros(swarm.dim)
                if np.any(too_close):
                    separation = -np.sum(diffs[too_close], axis=0) * self.separation
                
                # Apply rules
                swarm.velocities[i] += cohesion + alignment + separation
            
            # Limit speed
            speed = np.linalg.norm(swarm.velocities[i])
            if speed > MAX_SPEED:
                swarm.velocities[i] = (swarm.velocities[i] / speed) * MAX_SPEED
                
            # Update position
            swarm.positions[i] += swarm.velocities[i]

            # Update phase: the phase advances by speed * dt per frame, where dt is the
            # phase increment (0.0125). A phase reset to 0 marks a note onset, which the
            # PhaserMapping detects. Phases are not coupled between agents.
            dt = 0.0125
            swarm.phases[i] += np.linalg.norm(swarm.velocities[i]) * dt
            if swarm.phases[i] >= 1.0:
                swarm.phases[i] = 0.0

            # Boundary conditions, wrap around (not used)
            # swarm.positions[i] = np.mod(swarm.positions[i], [WIDTH, HEIGHT, DEPTH][:swarm.dim])

            # Boundary conditions: steer away from the edges when within edge_threshold,
            # then clamp the position to the world
            edge_threshold = 20.0
            steer_factor = 0.3
            if swarm.positions[i][0] < edge_threshold:
                swarm.velocities[i][0] += steer_factor
            elif swarm.positions[i][0] >= WIDTH - edge_threshold:
                swarm.velocities[i][0] -= steer_factor

            if swarm.positions[i][1] < edge_threshold:
                swarm.velocities[i][1] += steer_factor
            elif swarm.positions[i][1] >= HEIGHT - edge_threshold:
                swarm.velocities[i][1] -= steer_factor
                
            if swarm.dim == 3 and swarm.positions[i][2] < edge_threshold:
                swarm.velocities[i][2] += steer_factor
            elif swarm.dim == 3 and swarm.positions[i][2] >= DEPTH - edge_threshold:
                swarm.velocities[i][2] -= steer_factor

            swarm.positions[i][0] = np.clip(swarm.positions[i][0], 0, WIDTH)
            swarm.positions[i][1] = np.clip(swarm.positions[i][1], 0, HEIGHT)
            if swarm.dim == 3:
                swarm.positions[i][2] = np.clip(swarm.positions[i][2], 0, DEPTH)

# Alternative behaviour (not used in the experiments): a Boids-based predator-prey simulation
# where agent 0 is the predator and the rest are prey
class PredatorPrey(SwarmBehavior):
    def __init__(self, prey_walk=2.0, prey_sprint=3.5, pred_walk=2.0, pred_sprint=3.8, evasion_weight=0.8, chase_weight=0.1, panic_dist=60.0, strike_dist=80.0, vision=30.0, friction=0.96):
        # Regular Boids weights for the prey
        self.cohesion = COHESION_WEIGHT
        self.alignment = ALIGNMENT_WEIGHT
        self.separation = SEPARATION_WEIGHT
        self.sep_dist = SEPARATION_DIST
        self.vision = vision
        
        # Predator/Prey specific parameters
        self.pred_walk = pred_walk
        self.pred_sprint = pred_sprint
        self.pred_energy = 100.0 # Starts at 100%, stamina for sprints
        self.strike_dist = strike_dist # How close the predator needs to be to start a sprint

        self.prey_walk = prey_walk
        self.prey_sprint = prey_sprint

        self.evasion_weight = evasion_weight
        self.chase_weight = chase_weight
        self.panic_dist = panic_dist # How close the predator gets before prey scatters
        self.friction = friction # General speed reduction to prevent perpetual acceleration

    def update(self, swarm):
        for i in range(swarm.num_agents):
            pos = swarm.positions[i]
            vel = swarm.velocities[i]

            # Gradually reduce velocity to simulate friction and prevent perpetual acceleration
            swarm.velocities[i] *= self.friction
            
            # Compute distances to all other boids
            diffs = swarm.positions - pos
            dists = np.linalg.norm(diffs, axis=1)
            dists[i] = np.inf 
            
            # Agent 0 is the Predator, everyone else is Prey
            is_predator = (i == 0)
            
            if is_predator:
                # --- PREDATOR LOGIC ---
                # Find the closest prey to chase
                prey_dists = dists[1:] # Ignore index 0 (self)
                closest_prey_idx = np.argmin(prey_dists) + 1 # +1 to offset the slice
                min_dist_to_prey = dists[closest_prey_idx]
                min_allowed_speed = self.pred_walk
                
                chase_vector = swarm.positions[closest_prey_idx] - pos
                swarm.velocities[i] += chase_vector * self.chase_weight

                # Check if prey is close enough to strike and have energy
                if min_dist_to_prey < self.strike_dist and self.pred_energy > 0:
                    max_allowed_speed = self.pred_sprint
                    self.pred_energy -= 2.0  # Burn energy during the sprint
                else:
                    max_allowed_speed = self.pred_walk
                    if self.pred_energy < 100.0:
                        self.pred_energy += 1.5  # Recover energy when not sprinting
                
            else:
                # --- PREY LOGIC ---
                # 1. Standard Flocking (but only look at other prey, ignore the predator)
                prey_mask = np.ones(swarm.num_agents, dtype=bool)
                prey_mask[0] = False # Don't flock with the predator
                prey_mask[i] = False # Don't flock with self
                
                neighbors = (dists < self.vision) & prey_mask

                min_allowed_speed = self.prey_walk
                
                if np.any(neighbors):
                    # Cohesion: Steer towards center of mass of neighbors
                    center_of_mass = np.mean(swarm.positions[neighbors], axis=0)
                    cohesion = (center_of_mass - pos) * self.cohesion
                    
                    # Alignment: Steer towards average velocity of neighbors
                    avg_vel = np.mean(swarm.velocities[neighbors], axis=0)
                    alignment = (avg_vel - vel) * self.alignment
                    
                    # Separation: Steer away from boids that are too close
                    too_close = (dists < self.sep_dist) & prey_mask
                    separation = np.zeros(swarm.dim)
                    if np.any(too_close):
                        separation = -np.sum(diffs[too_close], axis=0) * self.separation
                    
                    swarm.velocities[i] += cohesion + alignment + separation
                
                # 2. Evasion (Run from Predator 0)
                dist_to_pred = dists[0]
                if dist_to_pred < self.panic_dist:
                    # Steer strictly away from the predator's position
                    evasion_vector = pos - swarm.positions[0] 
                    # The closer the predator, the stronger the panic
                    panic_multiplier = 1.0 - (dist_to_pred / self.panic_dist) 
                    swarm.velocities[i] += evasion_vector * self.evasion_weight * panic_multiplier
                    # Scale max speed between walk and panic based on proximity
                    max_allowed_speed = self.prey_walk + ((self.prey_sprint - self.prey_walk) * panic_multiplier)
                else:
                    max_allowed_speed = self.prey_walk

            # Keep the speed between the minimum and maximum allowed speed
            speed = np.linalg.norm(swarm.velocities[i])

            # If any agent stops, nudge them to prevent division by zero
            if speed == 0:
                swarm.velocities[i] = (np.random.rand(swarm.dim) - 0.5) * 0.1 
                speed = np.linalg.norm(swarm.velocities[i])
            
            # Limit speed
            if speed > max_allowed_speed:
                swarm.velocities[i] = (swarm.velocities[i] / speed) * max_allowed_speed
            elif speed < min_allowed_speed:
                swarm.velocities[i] = (swarm.velocities[i] / speed) * min_allowed_speed
            
            # Update position
            swarm.positions[i] += swarm.velocities[i]

            # Boundary conditions, wrap around (not used)
            # swarm.positions[i] = np.mod(swarm.positions[i], [WIDTH, HEIGHT, DEPTH][:swarm.dim])

            # Boundary conditions: steer away from the edges, then clamp to the world
            edge_threshold = 20.0
            steer_factor = 0.3
            if swarm.positions[i][0] < edge_threshold:
                swarm.velocities[i][0] += steer_factor
            elif swarm.positions[i][0] >= WIDTH - edge_threshold:
                swarm.velocities[i][0] -= steer_factor

            if swarm.positions[i][1] < edge_threshold:
                swarm.velocities[i][1] += steer_factor
            elif swarm.positions[i][1] >= HEIGHT - edge_threshold:
                swarm.velocities[i][1] -= steer_factor
                
            if swarm.dim == 3 and swarm.positions[i][2] < edge_threshold:
                swarm.velocities[i][2] += steer_factor
            elif swarm.dim == 3 and swarm.positions[i][2] >= DEPTH - edge_threshold:
                swarm.velocities[i][2] -= steer_factor

            swarm.positions[i][0] = np.clip(swarm.positions[i][0], 0, WIDTH)
            swarm.positions[i][1] = np.clip(swarm.positions[i][1], 0, HEIGHT)
            if swarm.dim == 3:
                swarm.positions[i][2] = np.clip(swarm.positions[i][2], 0, DEPTH)

# # # # # # # # # # #
# 3. MUSICAL MAPPER #
# # # # # # # # # # #

# Base class for musical mapping strategies
class MappingStrategy:
    """Base class for all musical mapping algorithms."""
    def process(self, swarm, port, active_notes):
        raise NotImplementedError("Each mapping strategy must have a process method.")
    
    def get_colors(self, num_agents):
        """Optional method to get colors for visualization."""
        return ['blue'] * num_agents

# Alternative mapping (not used in the experiments): the first 3 boids play individual notes
# based on their position and velocity
class SoloistMapping(MappingStrategy):
    """Lets the first 3 individual boids play notes."""
    def process(self, swarm, port, active_notes):
        if not port: return

        for i in range(min(3, swarm.num_agents)): # Just play top 3 boids
            pos = swarm.positions[i]
            vel = swarm.velocities[i]
            
            # Parameter Mapping
            # 1. Y-Axis -> Pitch
            raw_pitch = map_value(pos[1], 0, HEIGHT, MIN_NOTE, MAX_NOTE)
            pitch = get_closest_scale_note(raw_pitch, PITCH_SCALE)
            
            # 2. Speed -> MIDI velocity (loudness)
            speed = np.linalg.norm(vel)
            volume = int(map_value(speed, 0, MAX_SPEED, 40, 127))
            
            # 3. X-Axis -> Panning (MIDI CC 10)
            panning = int(map_value(pos[0], 0, WIDTH, 0, 127))
            # Send CC for Panning
            port.send(mido.Message('control_change', channel=0, control=10, value=panning))
            
            # Note On / Note Off logic
            if i in active_notes and active_notes[i] != pitch:
                # Turn off old note
                port.send(mido.Message('note_off', channel=0, note=active_notes[i], velocity=0))
            
            if i not in active_notes or active_notes[i] != pitch:
                # Turn on new note
                port.send(mido.Message('note_on', channel=0, note=pitch, velocity=volume))
                polarity = get_global_polarity(swarm)
                current_time = time.perf_counter()
                relative_time = current_time - run_starttime
                midi_event_timestamps.append({
                    'timestamp': relative_time,
                    'polarity': polarity
                    })
                active_notes[i] = pitch

    def get_colors(self, num_agents):
        # The 3 soloists get distinct colors, the rest are muted
        colors = ['gray'] * num_agents
        if num_agents > 0: colors[0] = 'crimson'
        if num_agents > 1: colors[1] = 'gold'
        if num_agents > 2: colors[2] = 'deepskyblue'
        return colors

# Alternative mapping (not used in the experiments): aggregates the entire swarm into a single
# voice that plays a two-note chord based on the swarm's overall state
class HiveMindMapping(MappingStrategy):
    """Treats the swarm as a single entity playing an aggreageted chord."""
    def process(self, swarm, port, active_notes):
        if not port: return
        
        # Macro-Mathematics
        # Center of mass (average position) and average velocity
        com = np.mean(swarm.positions, axis=0)
        avg_vel = np.mean(swarm.velocities, axis=0)
        
        # Density (Standard deviation of positions). Lower = tighter swarm.
        spread = np.mean(np.std(swarm.positions, axis=0)) 
        
        # Parameter Mapping
        # Y-Axis -> Pitch (Play a two-note chord)
        raw_pitch = map_value(com[1], 0, HEIGHT, MIN_NOTE, MAX_NOTE)
        root_note = get_closest_scale_note(raw_pitch, PITCH_SCALE)
        fifth_note = get_closest_scale_note(raw_pitch + 7, PITCH_SCALE)
        desired_notes = [root_note, fifth_note]
        
        # Speed -> MIDI velocity (loudness)
        speed = np.linalg.norm(avg_vel)
        volume = int(map_value(speed, 0, MAX_SPEED, 40, 127))
        
        # X-Axis -> Panning (CC 10)
        panning = int(map_value(com[0], 0, WIDTH, 0, 127))
        port.send(mido.Message('control_change', channel=0, control=10, value=panning))
        
        # Density -> Filter Brightness/Cutoff (CC 74)
        # Tighter swarm = darker sound; spread out = brighter sound
        brightness = int(map_value(spread, 0, WIDTH/2, 20, 127))
        port.send(mido.Message('control_change', channel=0, control=74, value=brightness))
        
        # 3D Mapping -> Depth controls Reverb (CC 91)
        if swarm.dim == 3:
            reverb = int(map_value(com[2], 0, DEPTH, 0, 127))
            port.send(mido.Message('control_change', channel=0, control=91, value=reverb))
            
        # Turn off notes that are no longer in our desired chord
        current_playing = active_notes.get(0, [])
        for note in current_playing:
            if note not in desired_notes:
                port.send(mido.Message('note_off', channel=0, note=note, velocity=0))
                
        # Turn on new notes
        new_playing = []
        for note in desired_notes:
            if note not in current_playing:
                port.send(mido.Message('note_on', channel=0, note=note, velocity=volume))
                polarity = get_global_polarity(swarm)
                current_time = time.perf_counter()
                relative_time = current_time - run_starttime
                midi_event_timestamps.append({
                    'timestamp': relative_time,
                    'polarity': polarity
                    })
            new_playing.append(note)
            
        active_notes[0] = new_playing

    def get_colors(self, num_agents):
        # Universal color for the hive mind, since it's a single entity
        colors = ['purple'] * num_agents
        if num_agents > 0:
            colors[0] = 'crimson'
        return colors

# Agent-centric parameter mapping used in the experiments. Each agent is a monophonic voice.
# When an agent's phase resets, it plays a note with pitch given by its x-coordinate (linear,
# truncated to an integer MIDI note) and MIDI velocity given by its speed. Notes are switched
# off after NOTE_DURATION seconds, or when the same agent triggers a new note.
class PhaserMapping(MappingStrategy):
    def process(self, swarm, port, active_notes):
        if not port: return

        current_time = time.perf_counter()
        # Turn off notes that have exceeded their duration
        for i in list(active_notes.keys()):
            note, start_time = active_notes[i]
            if current_time - start_time > NOTE_DURATION:
                port.send(mido.Message('note_off', note=note, velocity=0))
                del active_notes[i]

        # Agents whose phase was reset in this frame trigger a new note
        triggered_agents = np.where(swarm.phases == 0.0)[0]
        for i in triggered_agents:
            
            pos = swarm.positions[i]
            vel = swarm.velocities[i]

            # Map X-Axis -> Pitch
            pitch = int(map_value(pos[0], 0, WIDTH, MIN_NOTE, MAX_NOTE))

            # Map Speed -> MIDI velocity (loudness)
            speed = np.linalg.norm(vel)
            volume = int(map_value(speed, 0, MAX_SPEED, 40, 127))

            # Turn off old note if it's still active
            if i in active_notes:
                old_note, _ = active_notes[i]
                port.send(mido.Message('note_off', note=old_note, velocity=0))
            
            # Turn on new note and add it to active notes with timestamp
            port.send(mido.Message('note_on', note=pitch, velocity=volume))
            
            polarity = get_global_polarity(swarm)
            current_time = time.perf_counter()
            relative_time = current_time - run_starttime
            midi_event_timestamps.append({
                'timestamp': relative_time,
                'polarity': polarity
                })
            active_notes[i] = (pitch, current_time)


class MusicMapper:
    def __init__(self, strategy):
        self.strategy = strategy
        try:
            # Opens the virtual MIDI output port used in the experiments
            self.port = mido.open_output('loopMIDI Port 1') # Change to your MIDI port name
            print(f"Connected to MIDI Port: {self.port.name}")
        except OSError:
            print("WARNING: No MIDI port found. Running visual simulation only.")
            self.port = None

        self.active_notes = {} # Keep track of playing notes to turn them off

    def set_strategy(self, new_strategy):
        """Allows swapping the musical mapping on the fly."""
        # Turn off all currently playing notes before switching
        if self.port:
            for agent_id, notes in self.active_notes.items():
                if isinstance(notes, list): # Handle list of notes
                    for n in notes: self.port.send(mido.Message('note_off', channel=0, note=n, velocity=0))
                else: # Handle single note
                    self.port.send(mido.Message('note_off', channel=0, note=notes, velocity=0))
        self.active_notes.clear()
        self.strategy = new_strategy

    def process_swarm(self, swarm):
        self.strategy.process(swarm, self.port, self.active_notes)

    def get_active_notes(self):
        return self.active_notes

# # # # # # # # # # # # #
# 4. VISUALIZATION LOOP #
# # # # # # # # # # # # #

# Interactive mode: one frame of physics, sonification and data extraction, followed by rendering
def update_frame(frame):

    swarm.update()
    mapper.process_swarm(swarm)

    update_simulation_data(swarm, mapper.active_notes)

    colors = mapper.strategy.get_colors(swarm.num_agents)
    scatter.set_color(colors)
    
    if swarm.dim == 3:
        scatter._offsets3d = (swarm.positions[:,0], swarm.positions[:,1], swarm.positions[:,2])
    else:
        scatter.set_offsets(swarm.positions)
    
    return scatter

# # # # # # # # # # # # # # # # #
# 5. CONFIGURATION & PARAMETERS #
# # # # # # # # # # # # # # # # #
NUM_AGENTS = 20
DIMENSIONS = 3 # Change to 2 for 2D simulation, 3 for 3D
WIDTH, HEIGHT, DEPTH = 500, 500, 500
MAX_SPEED = 2.0

# Boids Algorithm Weights
COHESION_WEIGHT = 0.01 # Default is 0.01
ALIGNMENT_WEIGHT = 0.06 # Default is 0.06
SEPARATION_WEIGHT = 0.3 # Default is 0.3
SEPARATION_DIST = 20.0 # Ratio between vision and separation distance is 2.5:1
VISION = round(WIDTH*0.10, 0) # Default is 10% of width, i.e. 50 units

# Predator-prey weights (alternative behaviour, not used in the experiments)
PREY_WALK = 2.0
PREY_SPRINT = 3.5
PRED_WALK = 2.0
PRED_SPRINT = 3.5
EVASION_WEIGHT = 0.8
CHASE_WEIGHT = 0.1
PANIC_DIST = 20.0
STRIKE_DIST = 80.0
FRICTION = 0.96

# MIDI Mapping Configuration
# Map: (Min Input, Max Input) -> (Min MIDI, Max MIDI)
PITCH_SCALE = [60, 62, 64, 65, 67, 69, 71, 72] # C major scale, used by the soloist and hive mind mappings only
MIN_NOTE, MAX_NOTE = 36, 96 # C2 to C7, chromatic
NOTE_DURATION = 0.5 # seconds

# Data containers
# Stores data points
simulation_data = []
# Stores MIDI timestamps for Inter-Onset Intervals (IOI)
midi_event_timestamps = []

# Behaviour Initialization
boids_algo = Boids(COHESION_WEIGHT, ALIGNMENT_WEIGHT, SEPARATION_WEIGHT, SEPARATION_DIST, VISION)
predator_prey_algo = PredatorPrey(PREY_WALK, PREY_SPRINT, PRED_WALK, PRED_SPRINT, EVASION_WEIGHT, CHASE_WEIGHT, PANIC_DIST, STRIKE_DIST, VISION, FRICTION)
# Mapping Strategy Initialization
soloist_music = SoloistMapping()
hive_mind_music = HiveMindMapping()
phase_based_music = PhaserMapping()

# Choose behaviour and mapping strategy (Boids and PhaserMapping were used in the experiments)
BEHAVIOUR = boids_algo
MAPPING_STRATEGY = phase_based_music

BATCH_RUN = False
run_starttime = 0

# Swarm Initialization and Music Mapper
swarm = Swarm(NUM_AGENTS, DIMENSIONS, BEHAVIOUR)
mapper = MusicMapper(MAPPING_STRATEGY)

# # # # # #
# 6. MAIN #
# # # # # #
MAX_FRAMES = 2100 # Frames per batch run, of which the first 300 are treated as warm-up
if BATCH_RUN:
    # The plot window will not take focus from the OS
    plt.rcParams['figure.raise_window'] = False
    
    # Parameter sweeps. The loop below runs one sweep at a time: to run another
    # sweep, replace alignment2_increments, alignment2_filenames and ALIGNMENT_WEIGHT in the
    # loop with the corresponding list, filenames and weight. The RQ2 sweep was run as two
    # parts, alignment_increments (0.00 to 0.5) and alignment2_increments (0.6 to 1.5).
    cohesion_increments = [0.00, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
    cohesion_filenames = ['RQ1_Cohesion_0.00', 'RQ1_Cohesion_0.005', 'RQ1_Cohesion_0.01', 'RQ1_Cohesion_0.015', 'RQ1_Cohesion_0.02', 'RQ1_Cohesion_0.03', 'RQ1_Cohesion_0.04', 'RQ1_Cohesion_0.05', 'RQ1_Cohesion_0.06', 'RQ1_Cohesion_0.07']

    alignment_increments = [0.00, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.25, 0.5]
    alignment_filenames = ['RQ2_Alignment_0.00', 'RQ2_Alignment_0.01', 'RQ2_Alignment_0.02', 'RQ2_Alignment_0.03', 'RQ2_Alignment_0.05', 'RQ2_Alignment_0.08', 'RQ2_Alignment_0.1', 'RQ2_Alignment_0.15', 'RQ2_Alignment_0.25', 'RQ2_Alignment_0.5']

    alignment2_increments = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    alignment2_filenames = ['RQ2_Alignment_0.6', 'RQ2_Alignment_0.7', 'RQ2_Alignment_0.8', 'RQ2_Alignment_0.9', 'RQ2_Alignment_1.0', 'RQ2_Alignment_1.1', 'RQ2_Alignment_1.2', 'RQ2_Alignment_1.3', 'RQ2_Alignment_1.4', 'RQ2_Alignment_1.5']

   
    separation_increments = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0]
    separation_filenames = ['RQ3_Separation_0.0', 'RQ3_Separation_0.1', 'RQ3_Separation_0.2', 'RQ3_Separation_0.4', 'RQ3_Separation_0.6', 'RQ3_Separation_0.8', 'RQ3_Separation_1.0', 'RQ3_Separation_1.5', 'RQ3_Separation_2.0', 'RQ3_Separation_3.0']

    run_starttime = time.perf_counter()
    repeats = 5
    for r in range(repeats):
        print("Batch " + str(r+1) + " in progress.")
        run_starttime = time.perf_counter()
        for i in range(len(alignment2_increments)):
            ALIGNMENT_WEIGHT = alignment2_increments[i]
            filename = "Batch" + str(r+1) + "_" + "Run" + str(i+1) + "_" + alignment2_filenames[i]

            # Fresh data containers, swarm and mapper for every run
            simulation_data = [] 
            midi_event_timestamps = []

            boids_algo = Boids(COHESION_WEIGHT, ALIGNMENT_WEIGHT, SEPARATION_WEIGHT, SEPARATION_DIST, VISION)
            
            swarm = Swarm(NUM_AGENTS, DIMENSIONS, boids_algo) 
            mapper = MusicMapper(MAPPING_STRATEGY)

            # Close plot window
            plt.close('all')

            # Initial plot of the agent positions
            plt.ion()
            fig = plt.figure()
            initial_colors = mapper.strategy.get_colors(NUM_AGENTS)
            if DIMENSIONS == 3:
                ax = fig.add_subplot(111, projection='3d')
                scatter = ax.scatter(swarm.positions[:,0], swarm.positions[:,1], swarm.positions[:,2], c=initial_colors)
                ax.set_xlim(0, WIDTH); ax.set_ylim(0, HEIGHT); ax.set_zlim(0, DEPTH)
            else:
                ax = fig.add_subplot(111)
                scatter = ax.scatter(swarm.positions[:,0], swarm.positions[:,1], c=initial_colors)
                ax.set_xlim(0, WIDTH); ax.set_ylim(0, HEIGHT)

            plt.show()

            try:
                # Main loop for batch run
                for frame in range(MAX_FRAMES):
                    
                    start_time = time.perf_counter()

                    # Same logic as update_frame(), but without FuncAnimation
                    swarm.update()
                    mapper.process_swarm(swarm)
                    update_simulation_data(swarm, mapper.active_notes)

                    # Update visuals
                    colors = mapper.strategy.get_colors(swarm.num_agents)
                    scatter.set_color(colors)
                    
                    if swarm.dim == 3:
                        scatter._offsets3d = (swarm.positions[:,0], swarm.positions[:,1], swarm.positions[:,2])
                    else:
                        scatter.set_offsets(swarm.positions)

                    # Event Loop
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()
                    
                    # Each frame has a nominal period of 50 ms: measure the computation time
                    # and sleep only for the remainder
                    elapsed = time.perf_counter() - start_time
                    wait_time = max(0.001, 0.05 - elapsed) # Aim for 50ms total
                    
                    # A tiny pause keeps the GUI alive
                    plt.pause(0.001) 
                    if wait_time > 0.001:
                        time.sleep(wait_time - 0.001)

                print("Run " + str(i+1) + " with weight=" + str(ALIGNMENT_WEIGHT) + " complete. Closing window...")

            finally:
                # Cleanup: runs after MAX_FRAMES frames, or if the run is interrupted (Ctrl+C)
                plt.close(fig) 
                save_session_data(filename)
                
                # Cleanup MIDI
                if mapper.port:
                    for agent_id, val in mapper.active_notes.items():
                        if isinstance(val, tuple):
                            mapper.port.send(mido.Message('note_off', channel=0, note=val[0], velocity=0))
                        elif isinstance(val, list):
                            for n in val: mapper.port.send(mido.Message('note_off', channel=0, note=n, velocity=0))
                        else:
                            mapper.port.send(mido.Message('note_off', channel=0, note=val, velocity=0))
                    mapper.port.close()
                    time.sleep(1)
else:
    # Set the scene and initial scatter plot
    fig = plt.figure()
    initial_colors = mapper.strategy.get_colors(NUM_AGENTS)
    if DIMENSIONS == 3:
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(swarm.positions[:,0], swarm.positions[:,1], swarm.positions[:,2], c=initial_colors)
        ax.set_xlim(0, WIDTH); ax.set_ylim(0, HEIGHT); ax.set_zlim(0, DEPTH)
    else:
        ax = fig.add_subplot(111)
        scatter = ax.scatter(swarm.positions[:,0], swarm.positions[:,1], c=initial_colors)
        ax.set_xlim(0, WIDTH); ax.set_ylim(0, HEIGHT)


    try:
        # Run the animation
        ani = FuncAnimation(fig, update_frame, interval=50, blit=False)
        plt.show()
    finally:
        filename = generate_filename()
        save_session_data(filename)
        plt.close(fig)
        # Cleanup MIDI on close
        if mapper.port:
            for agent_id, val in mapper.active_notes.items():
                # Value is a tuple: (pitch, timestamp)
                if isinstance(val, tuple):
                    note_to_off = val[0]
                    mapper.port.send(mido.Message('note_off', channel=0, note=note_to_off, velocity=0))
                # Value is a list of pitches
                elif isinstance(val, list):
                    for n in val:
                        mapper.port.send(mido.Message('note_off', channel=0, note=n, velocity=0))
                # Value is just the pitch integer
                else:
                    mapper.port.send(mido.Message('note_off', channel=0, note=val, velocity=0))
            
            mapper.port.close()