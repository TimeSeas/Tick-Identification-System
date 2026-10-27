import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_species_specialist import gate_metrics,select_threshold,validate_candidate
from evaluate_species_cascade import (RecordingSpecies,load_rows,calibration_rows,
    validate_policy_binding,validate_classifier_contract)


class SpecialistEvaluationTests(unittest.TestCase):
    def test_frozen_holdout_policy_binds_model_manifest_and_exact_profile(self):
        policy=dict(profile_sha256='profile',candidate_sha256='model',manifest_sha256='manifest')
        validate_policy_binding(policy,'profile','model','manifest')
        for values in (('changed','model','manifest'),('profile','changed','manifest'),
                       ('profile','model','changed')):
            with self.assertRaises(ValueError):validate_policy_binding(policy,*values)

    def test_classifier_split_and_class_order_contract_is_verified_before_inference(self):
        profile=dict(species=dict(split_sha256='s',thresholds=dict(a=50,b=50),region_thresholds=dict(a=50,b=50)),
                     insect=dict(split_sha256='i',thresholds=dict(tick=50)))
        validate_classifier_contract(profile,'s','i',['a','b'],['tick'])
        with self.assertRaises(ValueError):validate_classifier_contract(profile,'changed','i',['a','b'],['tick'])
        with self.assertRaises(ValueError):validate_classifier_contract(profile,'s','changed',['a','b'],['tick'])
        with self.assertRaises(ValueError):validate_classifier_contract(profile,'s','i',['a','c'],['tick'])

    def test_recorded_probabilities_match_real_numpy_service_contract_and_cache(self):
        classifier=RecordingSpecies.__new__(RecordingSpecies)
        classifier.cache={};classifier.records=[];classifier.model=None
        classifier.input_size=448;classifier.device='cpu';classifier.view_mode='center_full'
        classifier.classes=['a','b']
        image=np.zeros((4,5,3),dtype=np.uint8)
        with patch('evaluate_species_cascade.probabilities',return_value=np.array([.8,.2])) as infer:
            self.assertEqual(classifier.predict(image)['slug'],'a')
            classifier.predict(image)
        self.assertEqual(infer.call_count,1)
        self.assertEqual(classifier.records,[[.8,.2],[.8,.2]])

    def test_specialist_cascade_population_contains_each_full_image_once(self):
        row=dict(path='a',bbox=None,source_species='species')
        data=dict(val=[row,dict(row,bbox=[0,0,2,2])])
        self.assertEqual(len(load_rows(data,'specialist','val')),1)
        data['val'].append(dict(row))
        with self.assertRaises(ValueError):load_rows(data,'specialist','val')

    def test_unknown_acceptance_is_per_source_image_and_any_view(self):
        rows=[dict(path='u',label=-1,probabilities=[.9,.1]),
              dict(path='u',label=-1,probabilities=[.4,.6]),
              dict(path='v',label=-1,probabilities=[.4,.6]),
              dict(path='k',label=0,probabilities=[.95,.05])]
        metrics=gate_metrics(rows,.8)
        self.assertEqual(metrics['unknown_images'],2)
        self.assertEqual(metrics['unknown_accepted_images'],1)
        self.assertEqual(metrics['unknown_false_acceptance'],.5)

    def test_threshold_jointly_constrains_unknowns_and_known_accuracy(self):
        rows=[dict(path=f'k{i}',label=0,probabilities=[.91,.09]) for i in range(20)]
        rows += [dict(path='wrong',label=0,probabilities=[.12,.88]),
                 dict(path='u',label=-1,probabilities=[.85,.15])]
        result=select_threshold(rows)
        self.assertAlmostEqual(result['threshold'],.91)
        self.assertEqual(result['metrics']['known_correct_accepted'],20)
        self.assertEqual(result['metrics']['unknown_accepted_images'],0)
        self.assertIsNone(select_threshold([dict(path='u',label=-1,probabilities=[.8,.2])]))

    def test_reject_all_is_not_a_successful_calibration(self):
        rows=[dict(path='k',label=0,probabilities=[.6,.4]),
              dict(path='u',label=-1,probabilities=[.95,.05])]
        self.assertIsNone(select_threshold(rows))

    def test_no_detection_unknown_is_in_denominator_but_never_accepted(self):
        rows=[dict(path='missing',label=-1,probabilities=[0.,0.],detected=False)]
        result=gate_metrics(rows,0.)
        self.assertEqual(result['unknown_images'],1)
        self.assertEqual(result['unknown_accepted_images'],0)

    def test_missing_known_image_remains_in_coverage_denominator(self):
        rows=calibration_rows(dict(path='missing',label='a'),[],['a','b'])
        rows.append(dict(path='found',label=0,probabilities=[.9,.1]))
        result=gate_metrics(rows,.8)
        self.assertEqual(result['known_images'],2)
        self.assertEqual(result['known_useful_coverage'],.5)

    def test_partial_ineligible_or_wrong_manifest_checkpoint_cannot_export(self):
        checkpoint=dict(partial=False,metrics=dict(eligible=True),
                        config=dict(manifest_sha256='abc'))
        validate_candidate(checkpoint,'abc')
        for change in (dict(partial=True),dict(metrics=dict(eligible=False)),
                       dict(config=dict(manifest_sha256='other'))):
            with self.assertRaises(ValueError):validate_candidate({**checkpoint,**change},'abc')


if __name__=='__main__':unittest.main()
