"""Run the noise comparison with one Slurm allocation per array/declination case.

`noise_pipeline.py` declares every case in one local recipe. Here the same
case declaration (`add_case`) forms one branch recipe whose inputs are the
values that differ between cases. A placed outer scatter gives each case its
own Slurm array task; inside that allocation the branch's imaging scatters run
locally.

This file imports `noise_pipeline`, so put this directory on PYTHONPATH and
pass it to `ninja compile --code-root` (see README.md).
"""

from __future__ import annotations

from noise_pipeline import CaseInputs, NoInputs, Settings, add_case, case_inputs
from shinobi import Recipe, SlurmPlacement, recipe_builder
from shinobi.steps.schema import InputRef


@recipe_builder(inputs_model=Settings)
def recipe(settings: Settings) -> Recipe:
    """Place each array/declination case in its own Slurm allocation."""
    branch = Recipe(name="noise-case", inputs_model=CaseInputs, outputs_model=NoInputs, max_workers=settings.max_workers)
    add_case(branch, settings, {field: InputRef(field=field) for field in CaseInputs.model_fields})

    cases = [case_inputs(settings, instrument, dec) for dec in settings.declinations for instrument in settings.arrays]
    columns = {field: [getattr(case, field) for case in cases] for field in CaseInputs.model_fields}
    pipeline = Recipe(name="array-noise", inputs_model=NoInputs, outputs_model=NoInputs, max_workers=settings.max_parallel)
    pipeline.add_step(
        "cases",
        branch,
        scatter=list(columns),
        placement=SlurmPlacement(sbatch_opts=settings.sbatch_opts, max_parallel=settings.max_parallel),
        **columns,
    )
    return pipeline
