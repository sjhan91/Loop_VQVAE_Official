import torch
import pretty_midi
import numpy as np


def check_time_sign(pm, num=4, denom=4):
    time_sign_list = pm.time_signature_changes
    
    # empty check
    if len(time_sign_list) == 0: 
        return False
    
    # nom and denom check
    for time_sign in time_sign_list:
        if time_sign.numerator != num or time_sign.denominator != denom:
            return False
        
    return True


def get_inst(inst_set):
    new_inst_set = []

    bass = np.arange(32, 40)
    drum_flag = False
    bass_flag = False
    
    for inst in inst_set:
        # check drum
        if inst.is_drum == True and drum_flag == False:
            new_inst_set.append(inst)
            drum_flag = True
        else:
            # check bass
            if inst.program in bass and bass_flag == False:
                new_inst_set.append(inst)
                bass_flag = True
    
    if drum_flag and bass_flag:
        return new_inst_set
    else:
        return None

    
def standarize_drum(pianoroll):
    standard_drum_set = [35, 38, 42, 45, 46, 48, 49, 50, 51]
    pianoroll[:, 35] = np.clip(np.sum(pianoroll[:, [35, 36]], axis=1), 0, 1) # kick
    pianoroll[:, 38] = np.clip(np.sum(pianoroll[:, [37, 38, 39, 40]], axis=1), 0, 1) # snare
    pianoroll[:, 42] = np.clip(np.sum(pianoroll[:, [42, 44]], axis=1), 0, 1) # closed hi-hat
    pianoroll[:, 45] = np.clip(np.sum(pianoroll[:, [41, 43, 45]], axis=1), 0, 1) # low tom
    pianoroll[:, 48] = np.clip(np.sum(pianoroll[:, [47, 48]], axis=1), 0, 1) # mid tom
    pianoroll[:, 49] = np.clip(np.sum(pianoroll[:, [49, 55, 57]], axis=1), 0, 1) # crash
    pianoroll[:, 51] = np.clip(np.sum(pianoroll[:, [51, 59]], axis=1), 0, 1) # ride
    
    return pianoroll[:, standard_drum_set] # 9 components


def play_pianoroll(pianoroll, fs):
    fs_time = 1 / fs
    standard_drum_set = [35, 38, 42, 45, 46, 48, 49, 50, 51]
    
    pm = pretty_midi.PrettyMIDI()
    bass = pretty_midi.Instrument(program=34, is_drum=False)
    drum = pretty_midi.Instrument(program=34, is_drum=True)
    
    pm.instruments.append(drum)
    pm.instruments.append(bass)
    
    time_tuple, pitch_tuple = np.where(pianoroll == 1)
    unique_pitch = np.unique(pitch_tuple)
    
    for pitch in unique_pitch:
        pitch_idx = np.where(pitch_tuple == pitch)[0]
        time_idx = time_tuple[pitch_idx]
        time_len = len(time_idx)
        
        
        if time_len > 1:
            if pitch < 9: # drum
                for i in range(time_len):
                    start_time = fs_time * (time_idx[i])
                    end_time = fs_time * (time_idx[i] + 1)
                    velocity = int(90 + np.round(2 * np.random.randn()))
                    drum.notes.append(pretty_midi.Note(velocity, standard_drum_set[pitch], start_time, end_time))
            else: # bass
                start_idx = time_idx[0]
                for i in range(time_len-1):
                    start_time = fs_time * start_idx
                    if time_idx[i]+1 != time_idx[i+1]:
                        end_time = (time_idx[i] + 1) * fs_time
                        velocity = int(80 + np.round(2 * np.random.randn()))
                        bass.notes.append(pretty_midi.Note(velocity, pitch+15, start_time, end_time))
                        start_idx = time_idx[i+1]

                end_time = (time_idx[i+1] + 1) * fs_time
                velocity = int(80 + np.round(2 * np.random.randn()))
                bass.notes.append(pretty_midi.Note(velocity, pitch+15, start_time, end_time))
                
    return pm


def remove_dup_step(pianoroll):
    dup_step = np.where(np.sum(pianoroll, axis=1) > 1)[0]
    
    for step in dup_step:
        dup_pitch = np.where(pianoroll[step] == 1)[0]
        pianoroll[step, dup_pitch[1:]] = 0
                   
    return pianoroll


def get_xor_corr(pianoroll):
    num_bar = 8
    num_note_per_bar = 16
    
    corr_list = []
    for i in range(num_bar):
        for j in range(i+1, num_bar):
            former = pianoroll[i*num_note_per_bar:(i+1)*num_note_per_bar]
            latter = pianoroll[j*num_note_per_bar:(j+1)*num_note_per_bar]
            corr_list.append(np.sum(np.logical_xor(former, latter)))
    
    # normalize
    corr_list = corr_list / (np.max(corr_list) + 1e-5)
    
    return 1 - corr_list