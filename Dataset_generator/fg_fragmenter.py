# -*- coding: utf-8 -*-
"""
Created on Tue June  5 20:29:17 2022
Last Updated: 08/20/2024

Fragmenter code for obtaining the functional group fragments of the components...

@author: Abhinav Abraham
"""

import vispy; import logging; vispy.use('pyqt5'); vispy_logger = logging.getLogger('vispy'); vispy_logger.setLevel(logging.CRITICAL)
from rdkit import Chem
import rdkit
import pandas as pd
import operator
import regex
from pathlib import Path
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class fragmenter:
    """
    Function to fragment and obtain the quantity of functional groups in the components...

    Müller, Simon. "Flexible heuristic algorithm for automatic molecule fragmentation: application to the UNIFAC group contribution model." Journal of cheminformatics 11 (2019): 1-12.

    Returns the functional group quantity of each component...

    """

    def function_to_choose_fragmentation(fragmentations):
        fragmentations_descriptors = {}
        i = 0
        for fragmentation in fragmentations:
            fragmentations_descriptors[i] = [len(fragmentation)]
            i += 1
        sorted_fragmentations_dict = sorted(fragmentations_descriptors.items(), key=operator.itemgetter(1))

        return fragmentations[sorted_fragmentations_dict[0][0]]

    def get_mol_from_SMARTS(smarts):
        mol = Chem.MolFromSmarts(smarts)
        mol.UpdatePropertyCache(strict = False)
        Chem.GetSymmSSSR(mol)

        return mol

    def get_mol_with_properties_from_SMARTS(SMARTS):
        mol_SMARTS = fragmenter.get_mol_from_SMARTS(SMARTS)
        SMARTS = SMARTS.replace('-,:', '')
        conditional_statement_exists = SMARTS.count(',') > 0
        if conditional_statement_exists:
            raise ValueError('Algorithm can\'t handle conditional SMARTS. Please use a list of SMARTS in the fragmentation scheme to mimic conditional SMARTS.')
        n_atoms_defining_SMARTS = 0
        n_available_bonds = 0
        n_atoms_with_available_bonds = 0
        n_hydrogens_this_molecule = 0
        n_atoms_in_ring = 0
        n_carbons = 0
        found_atoms = []
        for atom in mol_SMARTS.GetAtoms():
            SMARTS_atom = atom.GetSmarts()
            n_atoms_defining_SMARTS += 1
            n_available_bonds_this_atom = 0
            if atom.GetSymbol() == '*':
                matches = regex.finditer(r"\$?\(([^()]|(?R))*\)", SMARTS_atom)
                if matches is not None:
                    for match in matches:
                        m = match.group(0)
                        found_atoms.append(m[2:-1])
                        mol_SMARTS2 = fragmenter.get_mol_with_properties_from_SMARTS(m[2:-1])
                        n_atoms_defining_SMARTS += mol_SMARTS2.GetUnsignedProp('n_atoms_defining_SMARTS')
                        first_atom = mol_SMARTS2.GetAtomWithIdx(0)
                        n_hydrogens = first_atom.GetUnsignedProp('n_hydrogens')
                        n_available_bonds_this_atom = first_atom.GetTotalValence() - n_hydrogens
                        atom.SetUnsignedProp('n_hydrogens', n_hydrogens)
                        atom.SetUnsignedProp('n_available_bonds', n_available_bonds_this_atom)
                        if first_atom.GetAtomicNum() == 6:
                            n_carbons += 1
                    if len(found_atoms) == 1:
                        n_atoms_defining_SMARTS -= 1
                    elif len(found_atoms) > 1:
                        raise ValueError('Algorithm can\'t handle SMARTS with 2 levels of recursion')
                    else:
                        atom.SetUnsignedProp('n_hydrogens', 0)
                        atom.SetUnsignedProp('n_available_bonds', 0)
            else:
                n_available_bonds_this_atom = atom.GetImplicitValence()
                n_hydrogens = 0
                match = regex.findall(r'AtomHCount\s+(\d+)\s*', atom.DescribeQuery())
                if match:
                    n_hydrogens = int(match[0])
                n_available_bonds_this_atom -= n_hydrogens
                atom.SetUnsignedProp('n_hydrogens', n_hydrogens)
                if n_available_bonds_this_atom < 0:
                    n_available_bonds_this_atom = 0
                atom.SetUnsignedProp('n_available_bonds', n_available_bonds_this_atom)
                if atom.GetAtomicNum() == 6:
                    n_carbons += 1
                is_within_ring = False
                match = regex.findall(r'AtomInNRings\s+(-?\d+)\s+(!?=)\s*', atom.DescribeQuery())
                if match:
                     match = match[0]
                     n_rings = int(match[0])
                     comparison_sign = match[1]
                     if n_rings == -1:
                         is_within_ring = comparison_sign == '='
                     else:
                         if comparison_sign == '=':
                             is_within_ring  = n_rings != 0
                         else:
                             is_within_ring  = n_rings == 0
                if atom.GetIsAromatic() or is_within_ring:
                    n_atoms_in_ring += 1
            n_available_bonds +=n_available_bonds_this_atom
            n_hydrogens_this_molecule += atom.GetUnsignedProp('n_hydrogens')
            if n_available_bonds_this_atom > 0:
                n_atoms_with_available_bonds += 1
        atom_with_valence_one_on_carbon = Chem.MolFromSmarts('[*;v1][#6]')
        atom_with_valence_one_on_carbon.UpdatePropertyCache()
        atom_with_valence_one_on_excluding_carbon = Chem.MolFromSmarts('[$([*;v1][#6])]')
        atom_with_valence_one_on_excluding_carbon.UpdatePropertyCache()
        is_simple_atom_on_c = n_atoms_with_available_bonds == 1 and (mol_SMARTS.HasSubstructMatch(atom_with_valence_one_on_carbon) or mol_SMARTS.HasSubstructMatch(atom_with_valence_one_on_excluding_carbon))
        atom_with_valence_one = Chem.MolFromSmarts('[*;v1;!#1]')
        atom_with_valence_one.UpdatePropertyCache()
        is_simple_atom = n_atoms_with_available_bonds == 1 and (mol_SMARTS.HasSubstructMatch(atom_with_valence_one))
        if len(found_atoms) > 0:
            if len(found_atoms) == 1:
                sub_mol_SMARTS = Chem.MolFromSmarts(found_atoms[0])
                sub_mol_SMARTS.UpdatePropertyCache()
                is_simple_atom_on_c = (sub_mol_SMARTS.HasSubstructMatch(atom_with_valence_one_on_carbon) or sub_mol_SMARTS.HasSubstructMatch(atom_with_valence_one_on_excluding_carbon))
                is_simple_atom = n_atoms_defining_SMARTS == 1 and sub_mol_SMARTS.HasSubstructMatch(atom_with_valence_one)
            elif len(found_atoms) > 1:
                raise ValueError('Algorithm can\'t handle SMARTS with 2 recursive SMARTS')
        mol_SMARTS.SetUnsignedProp('n_hydrogens', n_hydrogens_this_molecule)
        mol_SMARTS.SetUnsignedProp('n_carbons', n_carbons)
        mol_SMARTS.SetUnsignedProp('n_available_bonds', n_available_bonds)
        mol_SMARTS.SetUnsignedProp('n_atoms_defining_SMARTS', n_atoms_defining_SMARTS)
        mol_SMARTS.SetUnsignedProp('n_atoms_with_available_bonds', n_atoms_with_available_bonds)
        mol_SMARTS.SetBoolProp('has_atoms_in_ring', n_atoms_in_ring > 0)
        mol_SMARTS.SetBoolProp('is_simple_atom', is_simple_atom)
        mol_SMARTS.SetBoolProp('is_simple_atom_on_c', is_simple_atom_on_c)
        mol_SMARTS.SetUnsignedProp('n_double_bonds', len(mol_SMARTS.GetSubstructMatches(Chem.MolFromSmarts("*=*"))))
        mol_SMARTS.SetUnsignedProp('n_triple_bonds', len(mol_SMARTS.GetSubstructMatches(Chem.MolFromSmarts("*#*"))))

        return mol_SMARTS

    def get_substruct_matches(mol_searched_for, mol_searched_in, atomIdxs_to_which_new_matches_have_to_be_adjacent, check_for_hydrogens = False):
        valid_matches = []
        if mol_searched_in.GetNumAtoms() >= mol_searched_for.GetNumAtoms():
            matches = mol_searched_in.GetSubstructMatches(mol_searched_for)
            if matches:
                for match in matches:
                    all_hydrogens_OK = True
                    if check_for_hydrogens:
                        for i in range(mol_searched_for.GetNumAtoms()):
                            atom_mol_searched_for = mol_searched_for.GetAtomWithIdx(i)
                            atom_mol_searched_in = mol_searched_in.GetAtomWithIdx(match[i])
                            if atom_mol_searched_in.HasProp('n_hydrogens'):
                                if atom_mol_searched_for.GetUnsignedProp('n_hydrogens') > atom_mol_searched_in.GetUnsignedProp('n_hydrogens'):
                                    all_hydrogens_OK = False
                                    break
                            else:
                                break
                    if all_hydrogens_OK:
                        add_this_match = True
                        if len(atomIdxs_to_which_new_matches_have_to_be_adjacent) > 0:
                            add_this_match = False
                            for i in match:
                                for neighbor in mol_searched_in.GetAtomWithIdx(i).GetNeighbors():
                                    if neighbor.GetIdx() in atomIdxs_to_which_new_matches_have_to_be_adjacent:
                                        add_this_match = True
                                        break
                        if add_this_match:
                            valid_matches.append(match)

        return valid_matches

    def get_heavy_atom_count(mol):
        heavy_atom_count = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 1:
                heavy_atom_count += 1

        return heavy_atom_count

    def __init__(self, fragmentation_scheme = {}, fragmentation_fg = None, algorithm = '', n_atoms_cuttoff = -1, function_to_choose_fragmentation = False, n_max_fragmentations_to_find = -1):
        if not type(fragmentation_scheme) is dict:
            raise TypeError('fragmentation_scheme must be a dctionary with integers as keys and either strings or list of strings as values.')
        if len(fragmentation_scheme) == 0:
            raise ValueError('fragmentation_scheme must be provided.')
        if not algorithm in ['simple', 'complete', 'combined']:
            raise ValueError('Algorithm must be either simple ,complete or combined.')
        if algorithm == 'simple':
            if n_max_fragmentations_to_find != -1:
                raise ValueError('Setting n_max_fragmentations_to_find only makes sense with complete or combined algorithm.')
        self.algorithm = algorithm
        if algorithm in ['complete', 'combined']:
            if n_atoms_cuttoff == -1:
                raise ValueError('n_atoms_cuttoff needs to be specified for complete or combined algorithms.')
            if function_to_choose_fragmentation == False:
                raise ValueError('function_to_choose_fragmentation needs to be specified for complete or combined algorithms.')
            if not callable(function_to_choose_fragmentation):
                raise TypeError('function_to_choose_fragmentation needs to be a function.')
            if n_max_fragmentations_to_find != -1:
                if n_max_fragmentations_to_find < 1:
                    raise ValueError('n_max_fragmentations_to_find has to be 1 or higher.')
        self.n_max_fragmentations_to_find = n_max_fragmentations_to_find
        self.n_atoms_cuttoff = n_atoms_cuttoff
        self.fragmentation_scheme = fragmentation_scheme
        self.function_to_choose_fragmentation = function_to_choose_fragmentation
        self._fragmentation_scheme_group_number_lookup = {}
        self._fragmentation_scheme_pattern_lookup = {}
        self.fragmentation_scheme_order = []
        self.fragmentation_fg = fragmentation_fg
        for group_number, list_SMARTS in fragmentation_scheme.items():
            self.fragmentation_scheme_order.append(group_number)
            if type(list_SMARTS) is not list:
                list_SMARTS = [list_SMARTS]
            for SMARTS in list_SMARTS:
                if SMARTS != '':
                    self._fragmentation_scheme_group_number_lookup[SMARTS] = group_number
                    mol_SMARTS = fragmenter.get_mol_with_properties_from_SMARTS(SMARTS)
                    self._fragmentation_scheme_pattern_lookup[SMARTS] = mol_SMARTS
        self._parent_pattern_lookup = {}
        for SMARTS1, mol_SMARTS1 in self._fragmentation_scheme_pattern_lookup.items():
            parent_patterns_of_SMARTS1 = []
            for SMARTS2, mol_SMARTS2 in self._fragmentation_scheme_pattern_lookup.items():
                if SMARTS1 != SMARTS2:
                    if mol_SMARTS2.GetNumAtoms() >= mol_SMARTS1.GetNumAtoms():
                        matches = fragmenter.get_substruct_matches(mol_SMARTS1, mol_SMARTS2, set(), True)
                        if matches:
                            parent_patterns_of_SMARTS1.append(SMARTS2)
            mol_SMARTS1.SetBoolProp('has_parent_pattern', len(parent_patterns_of_SMARTS1) > 0)
            self._parent_pattern_lookup[SMARTS1] = parent_patterns_of_SMARTS1

    def fragment(self, SMILES):
        is_valid_SMILES = Chem.MolFromSmiles(SMILES) is not None
        if not is_valid_SMILES:
            raise ValueError('Following SMILES is not valid: ' + SMILES)
        if SMILES.count('.') > 0:
            list_SMILES = SMILES.split('.')
        else:
            list_SMILES = [SMILES]
        success = False
        fragmentation = {}
        for SMILES in list_SMILES:
            temp_fragmentation, success = self.__get_fragmentation(SMILES)
            for SMARTS, matches in temp_fragmentation.items():
                group_number = self._fragmentation_scheme_group_number_lookup[SMARTS]
                if not self.fragmentation_fg[group_number-1] in fragmentation.keys():
                    fragmentation[self.fragmentation_fg[group_number-1]] = 0
                fragmentation[self.fragmentation_fg[group_number-1]] += len(matches)
            if not success:
                break

        return fragmentation, success

    def __get_fragmentation(self, SMILES):
        success = False
        fragmentation = {}
        if self.algorithm in ['simple', 'combined']:
            fragmentation, success = self.__simple_fragmentation(SMILES)
        if success:
            return fragmentation, success
        if self.algorithm in ['complete', 'combined']:
            fragmentations, success = self.__complete_fragmentation(SMILES)
            if success:
                fragmentation = self.function_to_choose_fragmentation(fragmentations)

        return fragmentation, success

    def __simple_fragmentation(self, SMILES):
        mol_SMILES = Chem.MolFromSmiles(SMILES)
        heavy_atom_count = fragmenter.get_heavy_atom_count(mol_SMILES)
        success = False
        fragmentation = {}
        fragmentation, atomIdxs_included_in_fragmentation = self.__search_non_overlapping_solution(mol_SMILES, {}, set(), set())
        success = len(atomIdxs_included_in_fragmentation) == heavy_atom_count
        level = 1
        while not success:
            fragmentation_so_far , atomIdxs_included_in_fragmentation_so_far = fragmenter.__clean_molecule_surrounding_unmatched_atoms(mol_SMILES, fragmentation, atomIdxs_included_in_fragmentation, level)
            level += 1
            if len(atomIdxs_included_in_fragmentation_so_far) == 0:
                break
            fragmentation_so_far, atomIdxs_included_in_fragmentation_so_far = self.__search_non_overlapping_solution(mol_SMILES, fragmentation_so_far, atomIdxs_included_in_fragmentation_so_far, atomIdxs_included_in_fragmentation_so_far)
            success = len(atomIdxs_included_in_fragmentation_so_far) == heavy_atom_count
            if success:
                fragmentation = fragmentation_so_far

        return fragmentation, success

    def __search_non_overlapping_solution(self, mol_searched_in, fragmentation, atomIdxs_included_in_fragmentation, atomIdxs_to_which_new_matches_have_to_be_adjacent, search_for_parent_patterns = True):
        n_atomIdxs_included_in_fragmentation = len(atomIdxs_included_in_fragmentation) - 1
        while n_atomIdxs_included_in_fragmentation != len(atomIdxs_included_in_fragmentation):
            n_atomIdxs_included_in_fragmentation = len(atomIdxs_included_in_fragmentation)
            for group_number in self.fragmentation_scheme_order:
                list_SMARTS = self.fragmentation_scheme[group_number]
                if type(list_SMARTS) is not list:
                    list_SMARTS = [list_SMARTS]
                for SMARTS in list_SMARTS:
                    if SMARTS != "":
                        fragmentation, atomIdxs_included_in_fragmentation = self.__get_next_non_overlapping_match(mol_searched_in, SMARTS, fragmentation, atomIdxs_included_in_fragmentation, atomIdxs_to_which_new_matches_have_to_be_adjacent, search_for_parent_patterns)

        return fragmentation, atomIdxs_included_in_fragmentation

    def __get_next_non_overlapping_match(self, mol_searched_in, SMARTS, fragmentation, atomIdxs_included_in_fragmentation, atomIdxs_to_which_new_matches_have_to_be_adjacent, search_for_parent_patterns):
        mol_searched_for = self._fragmentation_scheme_pattern_lookup[SMARTS]
        if search_for_parent_patterns:
            if mol_searched_for.GetBoolProp('has_parent_pattern'):
                for parent_SMARTS in self._parent_pattern_lookup[SMARTS]:
                    fragmentation, atomIdxs_included_in_fragmentation = self.__get_next_non_overlapping_match(mol_searched_in, parent_SMARTS, fragmentation, atomIdxs_included_in_fragmentation, atomIdxs_to_which_new_matches_have_to_be_adjacent, search_for_parent_patterns)
        if atomIdxs_to_which_new_matches_have_to_be_adjacent:
            matches = fragmenter.get_substruct_matches(mol_searched_for, mol_searched_in, atomIdxs_to_which_new_matches_have_to_be_adjacent)
        else:
            matches = fragmenter.get_substruct_matches(mol_searched_for, mol_searched_in, set())
        if matches:
            for match in matches:
                all_atoms_of_new_match_are_unassigned = atomIdxs_included_in_fragmentation.isdisjoint(match)
                if all_atoms_of_new_match_are_unassigned:
                    if not SMARTS in fragmentation:
                        fragmentation[SMARTS] = []
                    fragmentation[SMARTS].append(match)
                    atomIdxs_included_in_fragmentation.update(match)

        return fragmentation, atomIdxs_included_in_fragmentation

    def __clean_molecule_surrounding_unmatched_atoms(self, mol_searched_in, fragmentation, atomIdxs_included_in_fragmentation, level):
        for i in range(0, level):
            atoms_missing = set(range(0, fragmenter.get_heavy_atom_count(mol_searched_in))).difference(atomIdxs_included_in_fragmentation)
            new_fragmentation = fragmenter.__marshal.loads(fragmenter.__marshal.dumps(fragmentation))
            for atomIdx in atoms_missing:
                for neighbor in mol_searched_in.GetAtomWithIdx(atomIdx).GetNeighbors():
                    for smart, atoms_found in fragmentation.items():
                        for atoms in atoms_found:
                            if neighbor.GetIdx() in atoms:
                                if smart in new_fragmentation:
                                    if new_fragmentation[smart].count(atoms) > 0:
                                        new_fragmentation[smart].remove(atoms)
                        if smart in new_fragmentation:
                            if len(new_fragmentation[smart]) == 0:
                                new_fragmentation.pop(smart)
            new_atomIdxs_included_in_fragmentation = set()
            for i in new_fragmentation.values():
                for j in i:
                    new_atomIdxs_included_in_fragmentation.update(j)
            atomIdxs_included_in_fragmentation = new_atomIdxs_included_in_fragmentation
            fragmentation = new_fragmentation

        return fragmentation, atomIdxs_included_in_fragmentation

    def __complete_fragmentation(self, SMILES):
        mol_SMILES = self.__Chem.MolFromSmiles(SMILES)
        heavy_atom_count = fragmenter.get_heavy_atom_count(mol_SMILES)
        if heavy_atom_count > self.n_atoms_cuttoff:
            return {}, False
        completed_fragmentations = []
        groups_leading_to_incomplete_fragmentations = []
        completed_fragmentations, groups_leading_to_incomplete_fragmentations, incomplete_fragmentation_found = self.__get_next_non_overlapping_adjacent_match_recursively(mol_SMILES, heavy_atom_count, completed_fragmentations, groups_leading_to_incomplete_fragmentations, {}, set(), set(), self.n_max_fragmentations_to_find)
        success = len(completed_fragmentations) > 0

        return completed_fragmentations, success

    def __get_next_non_overlapping_adjacent_match_recursively(self, mol_searched_in, heavy_atom_count, completed_fragmentations, groups_leading_to_incomplete_fragmentations, fragmentation_so_far, atomIdxs_included_in_fragmentation_so_far, atomIdxs_to_which_new_matches_have_to_be_adjacent, n_max_fragmentations_to_find = -1):
        n_completed_fragmentations = len(completed_fragmentations)
        incomplete_fragmentation_found = False
        complete_fragmentation_found = False
        if len(completed_fragmentations) == n_max_fragmentations_to_find:
            return completed_fragmentations, groups_leading_to_incomplete_fragmentations, incomplete_fragmentation_found
        for group_number in self.fragmentation_scheme_order:
            list_SMARTS = self.fragmentation_scheme[group_number]
            if complete_fragmentation_found:
                break
            if type(list_SMARTS) is not list:
                list_SMARTS = [list_SMARTS]
            for SMARTS in list_SMARTS:
                if complete_fragmentation_found:
                    break
                if SMARTS != "":
                    matches = fragmenter.get_substruct_matches(self._fragmentation_scheme_pattern_lookup[SMARTS], mol_searched_in, atomIdxs_included_in_fragmentation_so_far)
                    for match in matches:
                        all_atoms_are_unassigned = atomIdxs_included_in_fragmentation_so_far.isdisjoint(match)
                        if not all_atoms_are_unassigned:
                            continue
                        for groups_leading_to_incomplete_fragmentation in groups_leading_to_incomplete_fragmentations:
                            if fragmenter.__is_fragmentation_subset_of_other_fragmentation(groups_leading_to_incomplete_fragmentation, fragmentation_so_far):
                                return completed_fragmentations, groups_leading_to_incomplete_fragmentations, incomplete_fragmentation_found
                        use_this_match = True
                        n_found_groups = len(fragmentation_so_far)
                        for completed_fragmentation in completed_fragmentations:
                            if not SMARTS in completed_fragmentation:
                                continue
                            if n_found_groups == 0:
                                use_this_match = not fragmenter.__is_match_contained_in_fragmentation(match, SMARTS, completed_fragmentation)
                            else:
                                if fragmenter.__is_fragmentation_subset_of_other_fragmentation(fragmentation_so_far, completed_fragmentation):
                                    use_this_match = not fragmenter.__is_match_contained_in_fragmentation(match, SMARTS, completed_fragmentation)
                            if not use_this_match:
                                break
                        if not use_this_match:
                            continue
                        this_SMARTS_fragmentation_so_far = fragmenter.__marshal.loads(fragmenter.__marshal.dumps(fragmentation_so_far))
                        this_SMARTS_atomIdxs_included_in_fragmentation_so_far = atomIdxs_included_in_fragmentation_so_far.copy()
                        if not SMARTS in this_SMARTS_fragmentation_so_far:
                            this_SMARTS_fragmentation_so_far[SMARTS] = []
                        this_SMARTS_fragmentation_so_far[SMARTS].append(match)
                        this_SMARTS_atomIdxs_included_in_fragmentation_so_far.update(match)
                        for groups_leading_to_incomplete_match in groups_leading_to_incomplete_fragmentations:
                            if fragmenter.__is_fragmentation_subset_of_other_fragmentation(groups_leading_to_incomplete_match, this_SMARTS_fragmentation_so_far):
                                use_this_match = False
                                break
                        if not use_this_match:
                            continue
                        if len(this_SMARTS_atomIdxs_included_in_fragmentation_so_far) < heavy_atom_count:
                            completed_fragmentations, groups_leading_to_incomplete_fragmentations, incomplete_fragmentation_found = self.__get_next_non_overlapping_adjacent_match_recursively(mol_searched_in, heavy_atom_count, completed_fragmentations, groups_leading_to_incomplete_fragmentations, this_SMARTS_fragmentation_so_far, this_SMARTS_atomIdxs_included_in_fragmentation_so_far, this_SMARTS_atomIdxs_included_in_fragmentation_so_far, n_max_fragmentations_to_find)
                            break
                        if len(this_SMARTS_atomIdxs_included_in_fragmentation_so_far) == heavy_atom_count:
                            completed_fragmentations.append(this_SMARTS_fragmentation_so_far)
                            complete_fragmentation_found = True
                            break
        if n_completed_fragmentations == len(completed_fragmentations):
            if not incomplete_fragmentation_found:
                incomplete_matched_groups = {}
                if len(atomIdxs_included_in_fragmentation_so_far) > 0:
                    unassignes_atom_idx = set(range(0, heavy_atom_count)).difference(atomIdxs_included_in_fragmentation_so_far)
                    for atom_idx in unassignes_atom_idx:
                        neighbor_atoms_idx = [i.GetIdx() for i in mol_searched_in.GetAtomWithIdx(atom_idx).GetNeighbors()]
                        for neighbor_atom_idx in neighbor_atoms_idx:
                            for found_smarts, found_matches in fragmentation_so_far.items():
                                for found_match in found_matches:
                                    if neighbor_atom_idx in found_match:
                                        if not found_smarts in incomplete_matched_groups:
                                            incomplete_matched_groups[found_smarts] = []
                                        if found_match not in incomplete_matched_groups[found_smarts]:
                                            incomplete_matched_groups[found_smarts].append(found_match)
                    is_subset_of_groups_already_found = False
                    indexes_to_remove = []
                    ind = 0
                    for groups_leading_to_incomplete_match in groups_leading_to_incomplete_fragmentations:
                        is_subset_of_groups_already_found = fragmenter.__is_fragmentation_subset_of_other_fragmentation(incomplete_matched_groups, groups_leading_to_incomplete_match)
                        if is_subset_of_groups_already_found:
                            indexes_to_remove.append(ind)
                        ind += 1
                    for index in sorted(indexes_to_remove, reverse=True):
                        del groups_leading_to_incomplete_fragmentations[index]
                    groups_leading_to_incomplete_fragmentations.append(incomplete_matched_groups)
                    groups_leading_to_incomplete_fragmentations = sorted(groups_leading_to_incomplete_fragmentations, key = len)
                    incomplete_fragmentation_found =  True

        return completed_fragmentations, groups_leading_to_incomplete_fragmentations, incomplete_fragmentation_found

    def __is_fragmentation_subset_of_other_fragmentation(self, fragmentation, other_fragmentation):
        n_found_groups = len(fragmentation)
        n_found_other_groups = len(other_fragmentation)
        if n_found_groups == 0:
            return False
        if n_found_other_groups < n_found_groups:
            return False
        n_found_SMARTS_that_are_subset = 0
        for found_SMARTS, found_matches in fragmentation.items():
            if found_SMARTS in other_fragmentation:
                found_matches_set = set(frozenset(i) for i in fragmentation[found_SMARTS])
                found_other_matches_set =  set(frozenset(i) for i in other_fragmentation[found_SMARTS])
                if found_matches_set.issubset(found_other_matches_set):
                    n_found_SMARTS_that_are_subset += 1
            else:
                return False

        return n_found_SMARTS_that_are_subset == n_found_groups

    def __is_match_contained_in_fragmentation(self, match, SMARTS, fragmentation):
        if not SMARTS in fragmentation:
            return False
        found_matches_set = set(frozenset(i) for i in fragmentation[SMARTS])
        match_set = set(match)

        return match_set in found_matches_set

class fg_from_smiles:
    """
    Function to find the create a file containing the components and its FG quantity...

    """

    def __init__(self, fg_prop_path, output_file_path, output_name, filepath_to_fuel_properties):
        self.output_file_path = output_file_path
        self.output_name = output_name
        self.fg_prop_path = fg_prop_path
        self.filepath_to_fuel_properties = filepath_to_fuel_properties
        fg_ids = pd.read_csv(self.fg_prop_path, header=0, index_col=None)
        self.fuel_details = pd.read_csv(self.filepath_to_fuel_properties, header=0, index_col=0)
        df_columns= ['Component']+fg_ids.fg_id.tolist()
        self.fg_df = pd.DataFrame(columns=df_columns)
        self.fg_df['Component'] = 'fg_wt'
        for num, fg in enumerate(fg_ids.fg_id.tolist()):
            self.fg_df.loc['fg_wt', fg] = fg_ids.fg_molwt[num]
        self.fg_df['Molecular Weight'] = 'N/A'

    def obtain_fg(self, mix_comps, mix_smiles):
        """
        Function to use rdkit and class fragmenter for obtaining the functional groups and their quantities for each component...

        Parameters
        ----------
        mix_comps : list
            A list of the component ids of components to be fragmeneted.
        mix_smiles : list
            A list of the SMILES of components to be fragmeneted.

        Raises
        ------
        Exception
            If rdkit version is below 2017.09.3 then an exception is raised to installed version 2017.09.3 or lower.

        Returns
        -------
        fg_df: dataframe
            A dataframe of functional groups and the quantities of each component.

        """

        if rdkit.rdBase.rdkitVersion < '2017.09.3':
            print('Installed rdkit version: ',rdkit.rdBase.rdkitVersion)
            raise Exception('This code has only been tested on version \'2017.09.3\'. Please install rdkit version 2017.09.3 or greater...')
        else:
            for sm1 in tqdm(mix_comps):
                fg_ids = pd.read_csv(self.fg_prop_path, header=0, index_col=None)
                if sm1.startswith('C'):
                    smiles = [str(mix_smiles.loc[mix_smiles.comp_id[mix_smiles.comp_id == str(sm1)].index.tolist()[0], 'SMILES'])]
                    mw_comp = [str(mix_smiles.loc[mix_smiles.comp_id[mix_smiles.comp_id == str(sm1)].index.tolist()[0], 'Molecular Weight'])]
                else: pass

                if type(smiles) is not list:
                    smiles = [smiles]
                else:
                    smiles = smiles
                fg_smarts = {str(fg_ids.loc[o,'UNIFAC_fg']):str(fg_ids.loc[fg_ids.UNIFAC_fg[fg_ids.UNIFAC_fg == str(str(fg_ids.loc[o,'UNIFAC_fg']))].index.tolist()[0], 'fg_smarts']) for o in range(len(fg_ids['UNIFAC_fg']))}
                fragmentation_scheme_list_enumerate = {i+1: j for i, j in enumerate(list(fg_smarts.values()))}
                pattern_descriptors = {}
                for group_number, SMARTS in fragmentation_scheme_list_enumerate.items():
                    if type(SMARTS) is list:
                        SMARTS = SMARTS[0]
                    if SMARTS != "":
                        pattern = fragmenter.get_mol_with_properties_from_SMARTS(SMARTS)
                        pattern_descriptors[group_number] = [pattern.GetUnsignedProp('n_available_bonds') == 0, (pattern.GetBoolProp('is_simple_atom_on_c') or pattern.GetBoolProp('is_simple_atom')),
                                                             pattern.GetUnsignedProp('n_atoms_defining_SMARTS'), pattern.GetUnsignedProp('n_available_bonds') == 1, fragmenter.get_heavy_atom_count(pattern) - pattern.GetUnsignedProp('n_carbons'),
                                                             pattern.GetBoolProp('has_atoms_in_ring'), pattern.GetUnsignedProp('n_triple_bonds'), pattern.GetUnsignedProp('n_double_bonds')]
                sorted_pattern_descriptors = sorted(pattern_descriptors.items(), key=operator.itemgetter(1), reverse=True)
                sorted_group_numbers = [i[0] for i in sorted_pattern_descriptors]
                combined_fragmenter = fragmenter(fragmentation_scheme_list_enumerate, list(fg_smarts.keys()), 'combined', 20, fragmenter.function_to_choose_fragmentation,1)
                combined_fragmenter.fragmentation_scheme_order = sorted_group_numbers
                i_structure = 0
                res = {}
                res_to_print = {}
                for SMILES in smiles:
                    i_structure = i_structure + 1
                    fragmentation, success = combined_fragmenter.fragment(SMILES)
                    n_heavy_atoms  = 0
                    for sub_SMILES in SMILES.split("."):
                        n_heavy_atoms = max(n_heavy_atoms, fragmenter.get_heavy_atom_count(Chem.MolFromSmiles(sub_SMILES)))
                    component_id = str(mix_smiles.loc[mix_smiles.SMILES[mix_smiles.SMILES == str(SMILES)].index.tolist()[0], 'comp_id'])

                    res[str(SMILES)] = list(fragmentation.items())
                    res_to_print[str(component_id)] = list(fragmentation.items())

                fg_ids = pd.read_csv(self.fg_prop_path, header=0, index_col=None)
                self.fg_df['Component'] = str(sm1)
                for fg in list(res_to_print.values())[0]:
                    fg_id = str(fg_ids.loc[fg_ids.UNIFAC_fg[fg_ids.UNIFAC_fg == str(fg[0])].index.tolist()[0], 'fg_id'])
                    self.fg_df.loc[str(sm1), fg_id] = int(fg[1])
                self.fg_df.loc[str(sm1), 'Molecular Weight'] = float(mw_comp[0])

        self.fg_df.Component = self.fg_df.index
        self.fg_df.index = list(range(len(self.fg_df)))
        self.fg_df.fillna(value=0, inplace=True)
        self.fg_df.drop(columns=[cc for cc in self.fg_df.columns[1:-1] if cc not in self.fuel_details.columns[:-2]], inplace=True)
        # self.fg_df = self.fg_df.loc[:, (self.fg_df.iloc[1:, :] != 0).any(axis=0)] # Drop all columns with only zeros...
        output_file_dir = os.path.join(self.output_file_path,self.output_name)
        self.fg_df.to_csv(output_file_dir, header=True, index=True, index_label='#')
        self.fg_df.set_index('Component', inplace=True)

        return self.fg_df

class execute:
    """
    Class for intitating the functional group quantity determination for each pure component in the palette...

    """

    def __init__(self, comp_details_path, fg_prop_path, filepath_to_fuel_properties):
        self.comp_details_path = comp_details_path
        self.output_file_path = Path(comp_details_path).parent
        self.output_name = 'Component_FG_information.csv'
        self.fg_prop_path = fg_prop_path
        self.filepath_to_fuel_properties = filepath_to_fuel_properties

    def perform(self):
        """
        Function to generate a dataframe of the functional group quantity of each pure component in the dataframe...

        Returns
        -------
        fg_df : dataframe
            A pandas dataframe of the pure components and its functional group compositions and other details.

        """
        comp_details = pd.read_csv(self.comp_details_path, header=0, index_col=0)
        comp_ids_manual = ['C'+str(int(1000000)+i)[1:] for i in range(1,len(comp_details)+1)]
        comp_details['comp_id'] = comp_ids_manual
        fg_class = fg_from_smiles(self.fg_prop_path, self.output_file_path, self.output_name, self.filepath_to_fuel_properties)
        fg_df = fg_class.obtain_fg(comp_ids_manual, comp_details)

        return fg_df