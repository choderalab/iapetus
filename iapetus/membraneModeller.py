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

class Modeller(app.Modeller):
    """
    Modify the Topology of a Modeller object to make MemProtMD_ lipid membranes
    compatibles with the all-atom AMBER force field.

    .. _MemProtMD: http://memprotmd.bioch.ox.ac.uk/home/

    Parameters
    ----------
    app.Modeller: OpenMM Modeller object

    """
    # TODO Support other membrane types
    # TODO Import atoms re-assigments and bonds types from XML files

    def modifyTopology(self):
        """
        Change atom and residue names to match AMBER types.

        Add bonds between non-hydrogen atoms. This is required since residues.xml,
        used by createStandardBonds()_ in OpenMM, does not contain bond definitions
        for lipid membranes.

        .. _createStandardBonds(): http://docs.openmm.org/latest/api-python/generated/simtk.openmm.app.topology.Topology.html#simtk.openmm.app.topology.Topology.createStandardBonds

        """

        atomsNames = {"N4":"N", "C1":"C13", "C2":"C14", "C3":"C15", "C5":"C12",
                      "C6":"C11", "P8":"P", "O9":"O13", "O10":"O14", "O11":"O11",
                      "O7":"O12", "C12":"C1", "C13":"C2", "O14":"O21", "C15":"C21",
                      "O16":"O22", "C32":"C3", "O33":"O31", "C34":"C31", "O35":"O32",
                      "C17":"C22", "C18":"C23", "C19":"C24", "C20":"C25" , "C21":"C26",
                      "C22":"C27", "C23":"C28", "C24":"C29", "C25":"C210", "C26":"C211",
                      "C27":"C212", "C28":"C213", "C29":"C214", "C30":"C215", "C31":"C216",
                      "C36":"C32", "C37":"C33", "C38":"C34", "C39":"C35", "C40":"C36",
                      "C41":"C37", "C42":"C38", "C43":"C39", "C44":"C310", "C45":"C311",
                      "C46":"C312", "C47":"C313", "C48":"C314", "C49":"C315", "C50":"C316"}

        residueBonds = [["C11", "O12"]]
        residueBonds += [["C12", "C11"]]
        residueBonds += [["C1", "C2"]]
        residueBonds += [["C1", "O11"]]
        residueBonds += [["C210", "C211"]]
        residueBonds += [["C211", "C212"]]
        residueBonds += [["C212", "C213"]]
        residueBonds += [["C213", "C214"]]
        residueBonds += [["C214", "C215"]]
        residueBonds += [["C215", "C216"]]
        residueBonds += [["C21", "C22"]]
        residueBonds += [["C21", "O22"]]
        residueBonds += [["C22", "C23"]]
        residueBonds += [["C23", "C24"]]
        residueBonds += [["C24", "C25"]]
        residueBonds += [["C25", "C26"]]
        residueBonds += [["C26", "C27"]]
        residueBonds += [["C27", "C28"]]
        residueBonds += [["C28", "C29"]]
        residueBonds += [["C29", "C210"]]
        residueBonds += [["C2", "C3"]]
        residueBonds += [["C2", "O21"]]
        residueBonds += [["C310", "C311"]]
        residueBonds += [["C311", "C312"]]
        residueBonds += [["C312", "C313"]]
        residueBonds += [["C313", "C314"]]
        residueBonds += [["C314", "C315"]]
        residueBonds += [["C315", "C316"]]
        residueBonds += [["C31", "C32"]]
        residueBonds += [["C31", "O32"]]
        residueBonds += [["C32", "C33"]]
        residueBonds += [["C33", "C34"]]
        residueBonds += [["C34", "C35"]]
        residueBonds += [["C35", "C36"]]
        residueBonds += [["C36", "C37"]]
        residueBonds += [["C37", "C38"]]
        residueBonds += [["C38", "C39"]]
        residueBonds += [["C39", "C310"]]
        residueBonds += [["C3", "O31"]]
        residueBonds += [["N", "C12"]]
        residueBonds += [["N", "C13"]]
        residueBonds += [["N", "C14"]]
        residueBonds += [["N", "C15"]]
        residueBonds += [["O12", "P"]]
        residueBonds += [["O21", "C21"]]
        residueBonds += [["O31", "C31"]]
        residueBonds += [["P", "O11"]]
        residueBonds += [["P", "O13"]]
        residueBonds += [["P", "O14"]]

        # Change atoms and residue names to match AMBER definitions:
        for residue in self.topology.residues():
            if residue.name == 'DPP':
                residue.name = 'DPPC'
                for atom in residue.atoms():
                    atom.name = atomsNames[atom.name]

        # Add bonds between non-hydrogen atoms:
        for residue in self.topology.residues():
            if residue.name == 'DPPC':
                atomList = {}
                for atom in residue.atoms():
                    atomList[atom.name] = atom
                for bond in residueBonds:
                    self.topology.addBond(atomList[bond[0]], atomList[bond[1]])

    def addHydrogens(self, forcefield=None, pH=7.0, variants=None, platform=None):
        """
        Call the Modeller's addHydrogens()_ method specifying also the hydrogens
        definitions for the lipid membrane.

        Load file myhydrogens.xml containing hydrogen definitions that should be
        used by addHydrogens()_.

        Returns
        -------

        Return same as addHydrogens()_.

        .. _addHydrogens(): http://docs.openmm.org/latest/api-python/generated/simtk.openmm.app.modeller.Modeller.html#simtk.openmm.app.modeller.Modeller.addHydrogens

        """
        self.loadHydrogenDefinitions(os.path.join(os.path.dirname(__file__), 'data/dppc', 'myhydrogens.xml'))
        self._hasLoadedStandardHydrogens = True
        return super(Modeller, self).addHydrogens(forcefield, pH, variants, platform)
