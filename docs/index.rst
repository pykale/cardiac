KaleCardiac
===========

.. image:: images/kalecardiac_logo_long.png
   :alt: KaleCardiac
   :align: center
   :width: 480px

Multimodal cardiovascular machine learning for the `PyKale <https://github.com/pykale/pykale>`_
ecosystem: ECG, chest X-ray, cardiac imaging and clinical data, with representation
learning and clinical prediction.

The package is a verb-oriented pipeline -- load, prepare, model, train, evaluate,
interpret -- and one page of the API reference covers each stage. The guides explain
how the stages fit together and what each model contributes; start with the quickstart.

**No data ships with KaleCardiac, and none ever will.** The studies it was refactored
from used private clinical cohorts. Every test and example runs on generated data,
which verifies that a workflow executes and is never a result.

.. toctree::
   :maxdepth: 2
   :caption: Guides

   quickstart
   architecture
   models
   multimodal_fusion
   migration_contract

.. toctree::
   :maxdepth: 2
   :caption: API reference

   kalecardiac
   kalecardiac.loaddata
   kalecardiac.prepdata
   kalecardiac.model
   kalecardiac.model.layers
   kalecardiac.model.embed
   kalecardiac.model.predict
   kalecardiac.pipeline
   kalecardiac.evaluate
   kalecardiac.interpret
   kalecardiac.utils
