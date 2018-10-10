# ==============================================================================
# FILE DOCSTRING
# ==============================================================================

"""

Membrane Modeller
=================

Add tools to the OpenMM Modeller_ to make MemProtMD_ coarse-grained membranes
compatibles with the all-atom AMBER force field.

.. _Modeller: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.app.modeller.Modeller.html#simtk.openmm.app.modeller.Modeller
.. _MemProtMD: http://memprotmd.bioch.ox.ac.uk/home/

"""

import os
import simtk.openmm.app as app

class MembraneModeller(app.Modeller):
    """
    Modify the Topology_ to make MemProtMD_ lipid membranes
    compatibles with the all-atom AMBER force field.

    Parameters
    ----------
    app.Modeller: OpenMM Modeller_ object

    .. _Topology: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.app.topology.Topology.html#simtk.openmm.app.topology.Topology
    .. _MemProtMD: http://memprotmd.bioch.ox.ac.uk/home/
    .. _Modeller: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.app.modeller.Modeller.html#simtk.openmm.app.modeller.Modeller

    """

    # TODO Support other membrane types
    # TODO Import atoms re-assigments and bonds types from XML files

    def modify_topology(self):
        """
        Change atom and residue names to match AMBER definitions.

        Add bonds between non-hydrogen atoms. This is required since residues.xml,
        used by createStandardBonds()_ in OpenMM, does not contain bond definitions
        for lipid membranes.

        .. _createStandardBonds(): http://docs.openmm.org/latest/api-python/generated/simtk.openmm.app.topology.Topology.html#simtk.openmm.app.topology.Topology.createStandardBonds

        """
        # atoms_names associates the names in the pdb file of the lipid membrane atoms
        # to its corresponding atom types as defined by the AMBER force field
        atom_names = {"N4": "N", "C1": "C13", "C2": "C14", "C3": "C15", "C5": "C12",
                      "C6": "C11", "P8": "P", "O9": "O13", "O10": "O14", "O11": "O11",
                      "O7": "O12", "C12": "C1", "C13": "C2", "O14": "O21", "C15": "C21",
                      "O16": "O22", "C32": "C3", "O33": "O31", "C34": "C31", "O35": "O32",
                      "C17": "C22", "C18": "C23", "C19": "C24", "C20": "C25" , "C21": "C26",
                      "C22": "C27", "C23": "C28", "C24": "C29", "C25": "C210", "C26": "C211",
                      "C27": "C212", "C28": "C213", "C29": "C214", "C30": "C215", "C31": "C216",
                      "C36": "C32", "C37": "C33", "C38": "C34", "C39": "C35", "C40": "C36",
                      "C41": "C37", "C42": "C38", "C43": "C39", "C44": "C310", "C45": "C311",
                      "C46": "C312", "C47": "C313", "C48": "C314", "C49": "C315", "C50": "C316"}

        # residue_bonds lists the bonds between non-hydrogen atoms in the lipid membrane
        residue_bonds = [["C11", "O12"]]
        residue_bonds += [["C12", "C11"]]
        residue_bonds += [["C1", "C2"]]
        residue_bonds += [["C1", "O11"]]
        residue_bonds += [["C210", "C211"]]
        residue_bonds += [["C211", "C212"]]
        residue_bonds += [["C212", "C213"]]
        residue_bonds += [["C213", "C214"]]
        residue_bonds += [["C214", "C215"]]
        residue_bonds += [["C215", "C216"]]
        residue_bonds += [["C21", "C22"]]
        residue_bonds += [["C21", "O22"]]
        residue_bonds += [["C22", "C23"]]
        residue_bonds += [["C23", "C24"]]
        residue_bonds += [["C24", "C25"]]
        residue_bonds += [["C25", "C26"]]
        residue_bonds += [["C26", "C27"]]
        residue_bonds += [["C27", "C28"]]
        residue_bonds += [["C28", "C29"]]
        residue_bonds += [["C29", "C210"]]
        residue_bonds += [["C2", "C3"]]
        residue_bonds += [["C2", "O21"]]
        residue_bonds += [["C310", "C311"]]
        residue_bonds += [["C311", "C312"]]
        residue_bonds += [["C312", "C313"]]
        residue_bonds += [["C313", "C314"]]
        residue_bonds += [["C314", "C315"]]
        residue_bonds += [["C315", "C316"]]
        residue_bonds += [["C31", "C32"]]
        residue_bonds += [["C31", "O32"]]
        residue_bonds += [["C32", "C33"]]
        residue_bonds += [["C33", "C34"]]
        residue_bonds += [["C34", "C35"]]
        residue_bonds += [["C35", "C36"]]
        residue_bonds += [["C36", "C37"]]
        residue_bonds += [["C37", "C38"]]
        residue_bonds += [["C38", "C39"]]
        residue_bonds += [["C39", "C310"]]
        residue_bonds += [["C3", "O31"]]
        residue_bonds += [["N", "C12"]]
        residue_bonds += [["N", "C13"]]
        residue_bonds += [["N", "C14"]]
        residue_bonds += [["N", "C15"]]
        residue_bonds += [["O12", "P"]]
        residue_bonds += [["O21", "C21"]]
        residue_bonds += [["O31", "C31"]]
        residue_bonds += [["P", "O11"]]
        residue_bonds += [["P", "O13"]]
        residue_bonds += [["P", "O14"]]

        # Change atoms and residue names to match AMBER definitions:
        for residue in self.topology.residues():
            if residue.name == 'DPP':
                residue.name = 'DPPC'
                for atom in residue.atoms():
                    atom.name = atom_names[atom.name]

        # Add bonds between non-hydrogen atoms:
        for residue in self.topology.residues():
            if residue.name == 'DPPC':
                atom_list = {}
                for atom in residue.atoms():
                    atom_list[atom.name] = atom
                for bond in residue_bonds:
                    self.topology.addBond(atom_list[bond[0]], atom_list[bond[1]])

    def addHydrogens(self, forcefield=None, pH=7.0, variants=None, platform=None):
        """
        Extend the Modeller's addHydrogens()_ method to add the missing hydrogens
        in both the protein and the membrane.

        Use the Modeller's loadHydrogenDefinitions_ method for loading the file
        data/dppc_hydrogens.xml, which contains the hydrogen definitions for the
        lipid membrane. The file hydrogen.xml, used by default in OpenMM,
        only contains information for the standard amino acids and nucleotides.

        Returns
        -------

        Return same as OpenMM Modeller's addHydrogens()_ method.

        .. _loadHydrogenDefinitions: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.app.modeller.Modeller.html#simtk.openmm.app.modeller.Modeller.loadHydrogenDefinitions
        .. _addHydrogens(): http://docs.openmm.org/latest/api-python/generated/simtk.openmm.app.modeller.Modeller.html#simtk.openmm.app.modeller.Modeller.addHydrogens

        """
        self.loadHydrogenDefinitions(os.path.join(os.path.dirname(__file__), 'data/dppc', 'dppc-hydrogens.xml'))
        return super().addHydrogens(forcefield, pH, variants, platform)
