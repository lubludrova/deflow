"""Check the selected manipulation configurations without creating simulators."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src/repro")]
import run_manipulation_stage1 as stage

COMMON = {'jac_sigma_target': 0.9, 'jac_warmup': 100000}

RECIPES = {('sac', 'sacsoft'): {'configuration': {'alpha_leak': 0.0,
                                        'alpha_min': 0.0,
                                        'alpha_prior': 0.2,
                                        'denoising_steps': 4,
                                        'ent_track': False,
                                        'gate_bias': 5.0,
                                        'grad_clip': 0.0,
                                        'lam_jac': 0.0,
                                        'logp_clip': 0.0,
                                        'lp_winsor': 0.0,
                                        'target_entropy_scale': 1.0,
                                        'u_scale': 3.0},
                      'actor_constructor': {'denoising_steps': 4}},
 ('sacflow', 'sacflowsoft'): {'configuration': {'alpha_leak': 0.0,
                                                'alpha_min': 0.0,
                                                'alpha_prior': 0.2,
                                                'denoising_steps': 4,
                                                'ent_track': False,
                                                'gate_bias': 5.0,
                                                'grad_clip': 0.0,
                                                'lam_jac': 0.0,
                                                'logp_clip': 0.0,
                                                'lp_winsor': 0.0,
                                                'target_entropy_scale': 0.0,
                                                'u_scale': 3.0},
                              'actor_constructor': {'denoising_steps': 4}},
 ('dime', 'dimesoft'): {'configuration': {'alpha_leak': 0.0,
                                          'alpha_min': 0.0,
                                          'alpha_prior': 0.2,
                                          'denoising_steps': 16,
                                          'ent_track': False,
                                          'gate_bias': 5.0,
                                          'grad_clip': 0.0,
                                          'lam_jac': 0.0,
                                          'logp_clip': 0.0,
                                          'lp_winsor': 0.0,
                                          'target_entropy_scale': 1.6647707349211722,
                                          'u_scale': 3.0},
                        'actor_constructor': {'denoising_steps': 16}},
 ('deflow', 'r1'): {'configuration': {'alpha_leak': 0.0,
                                      'alpha_min': 0.0,
                                      'alpha_prior': 0.05,
                                      'denoising_steps': 4,
                                      'ent_track': False,
                                      'gate_bias': 1.0,
                                      'grad_clip': 1.0,
                                      'lam_jac': 0.01,
                                      'logp_clip': 0.0,
                                      'lp_winsor': 0.0,
                                      'target_entropy_scale': 0.0,
                                      'u_scale': 4.0},
                    'actor_constructor': {'beta_bwd': 0.7,
                                          'beta_fwd': 0.7,
                                          'denoising_steps': 4,
                                          'density_estimator': 'exact',
                                          'exact_backward': True,
                                          'gate_bias': 1.0,
                                          'hidden_dim': 256,
                                          'integration': 'implicit',
                                          'jac_sigma_target': 0.9,
                                          'lam_bwd': 0.01,
                                          'lam_fwd': 0.01,
                                          'm_bwd': 5,
                                          'm_fwd': 5,
                                          'max_iter_bwd': 10,
                                          'max_iter_fwd': 25,
                                          'tol_bwd': 0.001,
                                          'tol_fwd': 1e-05,
                                          'u_scale': 4.0}},
 ('sac', 'sac'): {'configuration': {'alpha_leak': 0.0,
                                    'alpha_min': 0.0,
                                    'alpha_prior': 0.2,
                                    'denoising_steps': 4,
                                    'ent_track': False,
                                    'gate_bias': 5.0,
                                    'grad_clip': 0.0,
                                    'lam_jac': 0.0,
                                    'logp_clip': 0.0,
                                    'lp_winsor': 0.0,
                                    'target_entropy_scale': 1.0,
                                    'u_scale': 3.0},
                  'actor_constructor': {'denoising_steps': 4}},
 ('sacflow', 'sacflow'): {'configuration': {'alpha_leak': 0.0,
                                            'alpha_min': 0.0,
                                            'alpha_prior': 0.2,
                                            'denoising_steps': 4,
                                            'ent_track': False,
                                            'gate_bias': 5.0,
                                            'grad_clip': 0.0,
                                            'lam_jac': 0.0,
                                            'logp_clip': 0.0,
                                            'lp_winsor': 0.0,
                                            'target_entropy_scale': 0.0,
                                            'u_scale': 3.0},
                          'actor_constructor': {'denoising_steps': 4}},
 ('dime', 'dime'): {'configuration': {'alpha_leak': 0.0,
                                      'alpha_min': 0.0,
                                      'alpha_prior': 0.2,
                                      'denoising_steps': 16,
                                      'ent_track': False,
                                      'gate_bias': 5.0,
                                      'grad_clip': 0.0,
                                      'lam_jac': 0.0,
                                      'logp_clip': 0.0,
                                      'lp_winsor': 0.0,
                                      'target_entropy_scale': 1.6647707349211722,
                                      'u_scale': 3.0},
                    'actor_constructor': {'denoising_steps': 16}},
 ('deflow', 'tent0'): {'configuration': {'alpha_leak': 0.0,
                                         'alpha_min': 0.0,
                                         'alpha_prior': 0.05,
                                         'denoising_steps': 4,
                                         'ent_track': False,
                                         'gate_bias': 1.0,
                                         'grad_clip': 1.0,
                                         'lam_jac': 0.01,
                                         'logp_clip': 0.0,
                                         'lp_winsor': 0.0,
                                         'target_entropy_scale': 0.0,
                                         'u_scale': 4.0},
                       'actor_constructor': {'beta_bwd': 0.7,
                                             'beta_fwd': 0.7,
                                             'denoising_steps': 4,
                                             'density_estimator': 'exact',
                                             'exact_backward': True,
                                             'gate_bias': 1.0,
                                             'hidden_dim': 256,
                                             'integration': 'implicit',
                                             'jac_sigma_target': 0.9,
                                             'lam_bwd': 0.01,
                                             'lam_fwd': 0.01,
                                             'm_bwd': 5,
                                             'm_fwd': 5,
                                             'max_iter_bwd': 10,
                                             'max_iter_fwd': 25,
                                             'tol_bwd': 0.001,
                                             'tol_fwd': 1e-05,
                                             'u_scale': 4.0}},
 ('deflow', 'r2hw'): {'configuration': {'alpha_leak': 0.01,
                                        'alpha_min': 0.02,
                                        'alpha_prior': 0.05,
                                        'denoising_steps': 4,
                                        'ent_track': True,
                                        'gate_bias': 1.0,
                                        'grad_clip': 1.0,
                                        'lam_jac': 0.01,
                                        'logp_clip': 15.0,
                                        'lp_winsor': 0.05,
                                        'target_entropy_scale': 0.0,
                                        'u_scale': 4.0},
                      'actor_constructor': {'beta_bwd': 0.7,
                                            'beta_fwd': 0.7,
                                            'denoising_steps': 4,
                                            'density_estimator': 'exact',
                                            'exact_backward': True,
                                            'gate_bias': 1.0,
                                            'hidden_dim': 256,
                                            'integration': 'implicit',
                                            'jac_sigma_target': 0.9,
                                            'lam_bwd': 0.01,
                                            'lam_fwd': 0.01,
                                            'm_bwd': 5,
                                            'm_fwd': 5,
                                            'max_iter_bwd': 10,
                                            'max_iter_fwd': 25,
                                            'tol_bwd': 0.001,
                                            'tol_fwd': 1e-05,
                                            'u_scale': 4.0}}}

CELLS = [
    {"task": task, "method": method, "recipe": recipe,
     "configuration": {**COMMON, **RECIPES[(method, recipe)]["configuration"]},
     "actor_constructor": RECIPES[(method, recipe)]["actor_constructor"]}
    for task, method, recipe in [('mw_button', 'sac', 'sacsoft'),
 ('mw_button', 'sacflow', 'sacflowsoft'),
 ('mw_button', 'dime', 'dimesoft'),
 ('mw_button', 'deflow', 'r1'),
 ('ms_pickcube', 'sac', 'sac'),
 ('ms_pickcube', 'sacflow', 'sacflow'),
 ('ms_pickcube', 'dime', 'dime'),
 ('ms_pickcube', 'deflow', 'tent0'),
 ('ms_pushcube', 'sac', 'sac'),
 ('ms_pushcube', 'sacflow', 'sacflow'),
 ('ms_pushcube', 'dime', 'dime'),
 ('ms_pushcube', 'deflow', 'tent0'),
 ('mw_pushwall', 'sac', 'sacsoft'),
 ('mw_pushwall', 'sacflow', 'sacflowsoft'),
 ('mw_pushwall', 'dime', 'dimesoft'),
 ('mw_pushwall', 'deflow', 'r2hw'),
 ('mw_peg_native', 'sac', 'sacsoft'),
 ('mw_peg_native', 'sacflow', 'sacflowsoft'),
 ('mw_peg_native', 'dime', 'dimesoft'),
 ('mw_peg_native', 'deflow', 'r1')]
]


@pytest.mark.parametrize("cell", CELLS, ids=lambda c: c["task"] + "/" + c["method"])
def test_selected_recipe_and_actor(cell):
    args = stage.campaign.configure_method(cell["method"])
    for key, value in stage.RECIPES[cell["recipe"]].items():
        setattr(args, key, value)

    keys = {
        "denoising_steps", "u_scale", "gate_bias", "target_entropy_scale",
        "lam_jac", "jac_sigma_target", "jac_warmup", "ent_track", "alpha_leak",
        "alpha_prior", "lp_winsor", "logp_clip", "grad_clip", "alpha_min",
    }
    for key in keys:
        assert getattr(args, key) == cell["configuration"][key], key
    expected = {"sac": "gaussian_actor", "sacflow": "parent_flow_actor", "dime": "dime_dis_actor"}
    if cell["method"] == "deflow":
        actor_class = stage._gated_wrapper({})
    else:
        actor_class = stage.base.deq_multistep_flow_actor
        assert actor_class.__name__ == expected[cell["method"]]
    actor = actor_class(5, 2, -np.ones(2), np.ones(2), **cell["actor_constructor"])
    assert all(torch.isfinite(p).all() for p in actor.parameters())
    if cell["method"] == "deflow":
        assert actor.max_iter_fwd == 25 and actor.tol_fwd == 1e-5
        assert actor.exact_backward and actor.T == 4


def test_supported_scope():
    assert len(CELLS) == 20
    assert set(stage.RUN_ENVIRONMENTS) == {"mw_button", "mw_peg", "mw_pushwall", "ms_pickcube", "ms_pushcube"}
    assert set(stage.ACTOR_WRAPPERS) == {"gated"}
