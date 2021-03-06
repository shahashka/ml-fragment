import mdtraj as md
import numpy as np
from pymol import cmd
from pymol import stored
import matplotlib.pyplot as plt
import itertools
import warnings
import multiprocessing
from tqdm import tqdm
import cv2
import os
import warnings
warnings.filterwarnings('ignore')
import pickle
import pandas as pd
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-pdb', type=str, help='protein pdb input file')
    parser.add_argument('-csv', type=str, help='csv dock scores input file')
    parser.add_argument('-sdf', type=str, help='sdf input file')
    parser.add_argument('-sdftop', type=str, help='small sdf input file')
    parser.add_argument('-state', type=str, help='bottom or top 10k')
    args = parser.parse_args()
    return args

# pymol documentation
aa_hscale = {
            "ALA":  0.620,
            "ARG": -2.530,
            "ASN": -0.780,
            "ASP": -0.900,
            "CYS":  0.290,
            "GLN": -0.850,
            "GLU": -0.740,
            "GLY":  0.480,
            "HIS": -0.400,
            "ILE":  1.380,
            "LEU":  1.060,
            "LYS": -1.500,
            "MET":  0.640,
            "PHE":  1.190,
            "PRO":  0.120,
            "SER": -0.180,
            "THR": -0.050,
            "TRP":  0.810,
            "TYR":  0.260,
            "VAL":  1.080}

element_charge = {
           "H": +1, "I":-1,
           "N": -3,
           "P":-3,
           "C": +4,
           "O":-2,
           "F": -1,
           "S": -2,
           "Li": +1,
           "Cl": -1,
           "Br": -1}

element_radius = {
           "H": 1.20, "I": 1.98,
           "N": 1.55 ,
           "P":1.95,
           "C": 1.70,
           "O":1.52,
           "F": 1.47,
           "S":1.80,
           "Li": 1.82,
           "Cl": 1.75,
           "Br": 1.85}

def save_pocket(pdb, sdf, name):
    cmd.reinitialize()
    cmd.load(pdb)
    cmd.load(sdf, "LIG")
    cmd.select("POCK","br. LIG around 4")
    stored.idx = 0
    cmd.iterate("POCK and n. CA", expression="stored.idx+=1")
    cmd.save(name, "POCK")
    return stored.idx

# convert files into list of strings
# for the protein pdb and ligand sdf 
# (states are separated by $$$$)
def stringify(sdf, pdb, num_states, bottom):
    sdf_split = []
    with open(sdf,'r') as f:
        for key,group in itertools.groupby(f,lambda line: line.startswith('$$$$')):
          if not key:
            sdf_split.append("".join(list(group)))
    if bottom==True:
        sdf_split=sdf_split[-1*num_states:]
    else:
        sdf_split=sdf_split[0:num_states]
    with open(pdb,'r') as f:
        pdb_string = f.readlines()
    return sdf_split, "".join(pdb_string)

def find_end(lines):
    filtered_lines=[]
    for l in lines:
        split = l.split(' ')
        split = list(filter(lambda x: x != '', split))
        vals1 = (split[0])
        
        # stop when we get to the integers in the sdf
        if vals1.isdigit():
            return filtered_lines
        filtered_lines.append(l)
            
def create_pdb(sdf, pdb_string,i):
    lines = sdf.split('\n')
    lig = find_end(lines[4:])
    lig = list(filter(lambda x: 'H' not in x, lig))
    lig_pdb = list(map(create_line, lig, np.arange(1,len(lig)+1)))
    f =  open("tmp{0}.pdb".format(i), 'w+')
    f.write("\n".join(lig_pdb))
    f.write("\n")
    f.write(pdb_string)
    f.close()

# Create one line in the new pdb file for one state
# and the protein
# Use the standardized pdb regex to create the correct string
def create_line(line,i):
    split = line.split(' ')
    split = list(filter(lambda x: x != '', split))
    atom = split[3]
    vals1 = (float)(split[0])
    vals2 = (float)(split[1])
    vals3= (float)(split[2]) 
    
    # workaround to get the atoms in the ligand to be residues in mdtraj reader
    if i >=10 :
        new_string = "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}".format("HETATM",i,atom,'',"U"+str(i),'',0,'',vals1, vals2,vals3,0,0,atom,'')
    else:
        new_string = "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}".format("HETATM",i,atom,'',"UN"+str(i),'',0,'',vals1, vals2,vals3,0,0,atom,'')
    return new_string


# Create the data for ML model

# First array is contact matrix
# - ----------------| ligand atoms | protein residues
# - ligand atoms    |              | 
# ------------------|--------------|-----------------
# - protein residues|              |
#
# Second array is a mask for which cells correspond to ligand x ligand (-1), 
# protein x protein (1) and ligand x protein (0)
#
# Arrays are interpolated to 80x80
def maker_w_strings(data):
    i,sdf_state, num_residues, pdb = data
    tmp_file = str(id(multiprocessing.current_process()))
    create_pdb(sdf_state, pdb, tmp_file) # create the pdb for one state
    
    # Load the file into mdtraj
    t = md.load("tmp{0}.pdb".format(tmp_file))
    pl = len(list(t.topology.residues))
    resi = np.arange(pl)
    pairs = list(itertools.product(resi, resi))
    
    # compute constacts
    matrix, l = md.compute_contacts(t, contacts=pairs, scheme="closest-heavy")
    matrix = np.array(matrix).reshape((pl,pl))
            
    ### generate protein-ligand mask
    mask = np.zeros((pl,pl))
    for i in range(pl-num_residues):
        for j in range(pl-num_residues):
            # ligand
            mask[i,j] = -1
            
    for i in range(pl - num_residues,pl):
        for j in range( pl - num_residues,pl):
            # protein
            mask[i,j] = 1
            
    p_img = cv2.resize(matrix, dsize=(64, 64), interpolation=cv2.INTER_CUBIC).reshape(64,64,1)
    m_img = cv2.resize(mask, dsize=(64, 64), interpolation=cv2.INTER_CUBIC).reshape(64,64,1)

    p_contact_matrix = np.concatenate([p_img, m_img], axis=-1)
    os.remove("tmp{0}.pdb".format(tmp_file))
    return p_contact_matrix
    


def run():
    args = get_args()
    maps=[]

    protein = os.path.splitext(args.pdb)[0]
    num_residues = save_pocket(args.pdb, args.sdftop, protein +".pock.pdb")
    print("done creating pocket")
    s, pdb = stringify(args.sdf, protein+".pock.pdb", 10000, args.state=='bottom')
    pdb_string = [pdb for x in range(len(s))]

    with multiprocessing.Pool(32) as p:
        num_states = len(s)
        data = list(zip(range(num_states), s, num_residues*np.ones(num_states,dtype=int), pdb_string))
        itern = p.imap(maker_w_strings, data)
        for i,p_contact_matrix in enumerate(tqdm(itern, total=num_states)):
            maps.append(p_contact_matrix)
            
    print("Finished generating matrices")
    images = np.stack(maps,axis=0)
    
    # set the correct receptor
    scores = pd.read_csv(args.csv)
    scores["receptor"] = protein
    
    np.save(protein +"." + args.state+".matrices",images)
    scores.to_csv(protein + ".scores")

if __name__ == "__main__":
    run()   
