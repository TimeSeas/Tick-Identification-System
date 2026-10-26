"""Group integrity matters across tasks, not only inside one classifier."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_recognition_rebuild_data import Groups, assign_groups, assert_disjoint, near_matches


class RebuildDataTests(unittest.TestCase):
    def test_transitive_source_groups(self):
        groups = Groups()
        groups.union('sha_copy', 'photo_large')
        groups.union('photo_large', 'photo_original')
        groups.union('photo_original', 'same_observation')
        self.assertEqual(groups.find('sha_copy'), groups.find('same_observation'))

    def test_group_assignment_is_deterministic_and_each_class_has_holdouts(self):
        rows = [dict(task=t, label=label, group_id=f'{label}_{i}')
                for t in ('species', 'detector') for label in ('rare', 'common')
                for i in range(8)]
        first = assign_groups(rows)
        self.assertEqual(first, assign_groups(list(reversed(rows))))
        for label in ('rare', 'common'):
            self.assertEqual({first[f'{label}_{i}'] for i in range(8)},
                             {'train', 'val', 'calibration'})

    def test_cross_task_leakage_is_rejected(self):
        manifests = {'species': {'train': [{'group_id': 'x', 'sha256': 'a'}]},
                     'detector': {'val': [{'group_id': 'x', 'sha256': 'b'}]}}
        with self.assertRaises(AssertionError):
            assert_disjoint(manifests)

    def test_same_task_cross_split_sha_is_rejected(self):
        with self.assertRaises(AssertionError):
            assert_disjoint({'species': {'train': [{'group_id': 'x', 'sha256': 'a'}],
                                        'val': [{'group_id': 'y', 'sha256': 'a'}]}})

    def test_near_duplicate_screen_keeps_distance_limit(self):
        value = 0b1111
        index = {(0, value): {value, 0b11111111}}
        self.assertEqual(set(near_matches(value, index)), {value, 0b11111111})
        self.assertEqual(near_matches(value, {(0, value): {0xffff}}), [])


if __name__ == '__main__':
    unittest.main()
