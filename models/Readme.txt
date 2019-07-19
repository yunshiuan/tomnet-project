------------------------------------------
This folder contains scripts and training results for different versions of model
------------------------------------------

------------------------------------------
raw_model
------------------------------------------
(1) This is the model built by Edwinn and treats data as saparate steps instead of 
continuous trajectories.
(2) This model only includes charnet but not predent.
(3) The performance is really poor.

------------------------------------------
working_model
- test_on_simulation_data
	training result on simulation data (S002a)
- test_on_human_data
	training result on human data (S030)
------------------------------------------
(1) This is the working model built by Yun-Shiuan.
(2) The model treats data as trajectories rather than separate steps.
(3) The model includes both charnel and predent.
(3) The performance awesome. The model are trained separately for both simulation and human data.
	(1) For simulation data (S002a) with the model composed of both charnet and predent, the validation accuracy = 96.98%, testing accuracy = 97.48%
	(2) For simulation data (S002a) with the model composed of only charnet, the validation accuracy = 99.7%, testing accuracy = 99.7%
	(3) For human data (S030) with the model composed of both charnet and predent, the validation accuracy = ?%, testing accuracy = ?%
	(4) For human data (S030) with the model composed of only charnet, the validation accuracy = 99.9%, testing accuracy = 99.69%

------------------------------------------
sandbox
- temporary_testing_version
	The codes root from Edwinn's codes ('raw_model') with step-by-step modification.
------------------------------------------
(1) This folder contains models for testing that are no longer in use.


