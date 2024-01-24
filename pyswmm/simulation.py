# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2024 Bryant E. McDonnell (See AUTHORS)
#
# Licensed under the terms of the BSD2 License
# See LICENSE.txt for details
# -----------------------------------------------------------------------------
"""Base class for a SWMM Simulation."""

# Standard import
from warnings import warn

# Local imports
from pyswmm.swmm5 import PySWMM, PYSWMMException
from pyswmm.toolkitapi import SimulationTime, SimulationUnits
from pyswmm.errors import MultiSimulationError


class _SimulationStateManager(object):
    """This manager was created to be a guardrail for the PySWMM developers
    experience.  In the event the developer is unaware of the non thread-safe
    non-reenterant quality of USEPA-SWMM, this prevents the developer from trying
    to open multiple instances of SWMM inside one instance of Python.

    The State Manager also give the option to show a simulation progress bar for
    the users running this code on the command line."""

    def __init__(self):
        self._sim_is_instantiated = False

    @property
    def sim_is_instantiated(self) -> bool:
        return self._sim_is_instantiated

    @sim_is_instantiated.setter
    def sim_is_instantiated(self, val: bool) -> None:
        self._sim_is_instantiated = val


# Module level instance for the Simulation Manager.
_sim_state_instance = _SimulationStateManager()


class Simulation(object):
    """
    Base class for a SWMM Simulation.

    The model object provides several options to run a simulation.

    Initialize the Simulation class.

    :param str inpfile: Name of SWMM input file (default '')
    :param str rptfile: Report file to generate (default None)
    :param str binfile: Optional binary output file (default None)

    Examples:

    Intialize using with statement.  This automatically cleans up
    after a simulation

    .. code-block:: python

        from pyswmm import Simulation

        with Simulation('tests/data/model_weir_setting.inp') as sim:
            for step in sim:
                pass

    Initialize the simulation and execute. This style does not allow
    the user to interact with the simulation. However, this approach
    tends to be the fastest.

    .. code-block:: python

        from pyswmm import Simulation

        sim = Simulation('tests/data/model_weir_setting.inp')
        sim.execute()
    """

    def __init__(self,
                 inputfile,
                 reportfile=None,
                 outputfile=None):

        # Add Simulation State Manager to Prevent Multiple Instances of
        # SWMM to be opened in one instance of Python
        if _sim_state_instance.sim_is_instantiated:
            raise(MultiSimulationError("Multi-Simulation Error."))

        self._model = PySWMM(inputfile, reportfile, outputfile)
        self._model.swmm_open()
        self._is_open = True
        _sim_state_instance.sim_is_instantiated = self._is_open
        self._advance_seconds = None
        self._is_started = False
        self._terminate_request = False
        self._callbacks = {
            "before_start": None,
            "after_start": None,
            "before_step": None,
            "after_step": None,
            "before_end": None,
            "after_end": None,
            "after_close": None
        }

    def __enter__(self):
        """
        Examples:

        .. code-block:: python

            from pyswmm import Simulation

            with Simulation('tests/data/model_weir_setting.inp') as sim:
                for step in sim:
                    print(sim.current_time)

        .. code-block::

            >>> 2015-11-01 14:00:30
            >>> 2015-11-01 14:01:00
            >>> 2015-11-01 14:01:30
            >>> 2015-11-01 14:02:00

        """
        return self

    def __iter__(self):
        """Iterator over Simulation"""
        return self

    def start(self):
        """Start Simulation (no longer suggested to user)."""
        if not self._is_started:
            # Execute Callback Hooks Before Start
            self._execute_callback(self._before_start())
            self._model.swmm_start(True)
            # Execute Callback Hooks After Start
            self._execute_callback(self._after_start())
            self._is_started = True

    def __next__(self):
        """Next"""
        # Start Simulation
        self.start()
        # Check if simulation termination request was made
        if self._terminate_request:
            self._execute_callback(self._before_end())
            raise StopIteration
        # Execute Callback Hooks Before Simulation Step
        self._execute_callback(self._before_step())
        # Simulation Step Amount
        if self._advance_seconds is None:
            time = self._model.swmm_step()
        else:
            time = self._model.swmm_stride(self._advance_seconds)
        # Execute Callback Hooks After Simulation Step
        self._execute_callback(self._after_step())
        if time <= 0.0:
            self._execute_callback(self._before_end())
            raise StopIteration
        return self._model

    def __exit__(self, *a):
        """close"""
        if self._is_started:
            self._model.swmm_end()
            self._is_started = False
            # Execute Callback Hooks After Simulation End
            self._execute_callback(self._after_end())
        if self._is_open:
            self._model.swmm_close()
            self._is_open = False
            # Execute Callback Hooks After Simulation Closes
            self._execute_callback(self._after_close())
        _sim_state_instance.sim_is_instantiated = self._is_open

    @staticmethod
    def _is_callback(callable_object):
        """Checks if arugment is a function/method."""
        if not callable(callable_object):
            error_msg = 'Requires Callable Object, not {}'.format(
                type(callable_object))
            raise (PYSWMMException(error_msg))
        else:
            return True

    def _execute_callback(self, callback):
        """Runs the callback."""
        if callback:
            try:
                callback()
            except PYSWMMException:
                error_msg = "Callback Failed"
                raise PYSWMMException((error_msg))

    @property
    def _isOpen(self) -> bool:
        """._isOpen is set for Deprecation """
        warn('This method will be deprecated in PySWMM-v2.1',
             DeprecationWarning, stacklevel=2)
        return self.sim_is_open

    @property
    def sim_is_open(self) -> bool:
        """Check is Model is Open

        Examples:

        .. code-block:: python

            from pyswmm import Simulation

            with Simulation('tests/data/model_weir_setting.inp') as sim:
                print(sim.sim_is_open)

        .. code-block::

            >>> True
        """
        return self._is_open

    @property
    def _isStarted(self) -> bool:
        """._isSpen is set for Deprecation """
        warn('This method will be deprecated in PySWMM-v2.1',
             DeprecationWarning, stacklevel=2)
        return self.sim_is_started

    @property
    def sim_is_started(self) -> bool:
        """Check is Simulation is Started

        Examples:

        .. code-block:: python

            from pyswmm import Simulation

            with Simulation('tests/data/model_weir_setting.inp') as sim:
                print(sim.sim_is_started)
                for step in sim:
                    print(sim.sim_is_started)

        .. code-block::

            >>> False
            >>> True
        """
        return self._is_started

    def initial_conditions(self, init_conditions):
        """
        Starting in PySWMM-v2 this method/function is set to be
        deprecated.  For setting initial depths refer to the
        Simulation.add_before_start() callback. If the user's goal is to
        set the initial link settings, instead use Simulation.add_after_start().
        """
        warn('This method was deprecated in PySWMM-v2',
             DeprecationWarning, stacklevel=2)

    def step_advance(self, advance_seconds):
        """
        Advances the model by X number of seconds instead of
        intervening at every routing step.  This does not change
        the routing step for the simulation; only lets python take
        back control after each advance period.

        :param int advance_seconds: Seconds to Advance simulation

        Examples:

        .. code-block:: python

            from pyswmm import Simulation

            with Simulation('tests/data/model_weir_setting.inp') as sim:
                sim.step_advance(30)
                for step in sim:
                    print(sim.current_time)

        .. code-block::

            >>> 2015-11-01 14:00:30
            >>> 2015-11-01 14:01:00
            >>> 2015-11-01 14:01:30
            >>> 2015-11-01 14:02:00
        """
        self._advance_seconds = advance_seconds

    def terminate_simulation(self):
        """
        Inserts a request to stop a simulation and cleanly executing the callbacks.

        Examples:

        .. code-block:: python

            with Simulation("model") as sim:
                nodeXYZ = Nodes(sim)["nodeZYX"]

                def before_step_callback():
                    if nodeXYZ.depth > 8:
                        sim.terminate_simulation()

                # Setting Callbacks
                sim.add_before_step(before_step_callback)

                for ind, step in enumerate(sim):
                    # Now simulation will end early if the depth is > 8
                    pass
        """
        self._terminate_request = True

    def report(self):
        """
        Writes to report file after simulation (no longer suggested for user).

        Examples:

        .. code-block:: python

            from pyswmm import Simulation

            with Simulation('tests/data/model_weir_setting.inp') as sim:
                for step in sim:
                    pass
                sim.report()
        """
        self._model.swmm_report()

    def close(self):
        """
        Intialize a simulation and iterate through a simulation. This
        approach requires some clean up. No longer recommended that the user
        call this function directly.
        """
        self.__exit__()

    def execute(self):
        """
        Open an input file, run SWMM, then close the file.

        Examples:

        .. code-block:: python

            sim = Simulation('tests/data/model_weir_setting.inp')
            sim.execute()
        """
        self._model.swmmExec()
        # swmm exec brings the simulation to a close therefore we
        # need to tell the sim state manager that we are free to
        # open another a simulation.
        self._is_open = False
        _sim_state_instance.sim_is_instantiated = self._is_open

    @property
    def engine_version(self):
        """
        Retrieves the SWMM Engine Version.

        :return: Engine Version
        :rtype: LooseVersion

        Examples:

        .. code-block:: python

            with Simulation('tests/data/model_weir_setting.inp') as sim:
                print(sim.engine_version)

        .. code-block::

            >>> 5.1.14
        """
        return self._model.swmm_getVersion()

    @property
    def runoff_error(self):
        """
        Retrieves the Runoff Mass Balance Error.

        :return: Runoff Mass Balance Error
        :rtype: float

        Examples:

        .. code-block:: python

            with Simulation('tests/data/model_weir_setting.inp') as sim:
               sim.execute()
               print(sim.runoff_error)

        .. code-block::

            >>> 0.01
        """
        return self._model.swmm_getMassBalErr()[0]

    @property
    def flow_routing_error(self):
        """
        Retrieves the Flow Routing Mass Balance Error.

        :return: Flow Routing Mass Balance Error
        :rtype: float

        Examples:

        .. code-block:: python

            with Simulation('tests/data/model_weir_setting.inp') as sim:
               sim.execute()
               print(sim.flow_routing_error)

        .. code-block::

            >>> 0.01
        """
        return self._model.swmm_getMassBalErr()[1]

    @property
    def quality_error(self):
        """
        Retrieves the Quality Routing Mass Balance Error.

        :return: Quality Routing Mass Balance Error
        :rtype: float

        Examples:

        .. code-block:: python

            with Simulation('tests/data/model_weir_setting.inp') as sim:
               sim.execute()
               print(sim.quality_error)

        .. code-block::

            >>> 0.01
        """
        return self._model.swmm_getMassBalErr()[2]

    @property
    def start_time(self):
        """Get/set Simulation start time.

        Examples:

        .. code-block:: python

            from pyswmm import Simulation

            with Simulation('tests/data/model_weir_setting.inp') as sim:
                print(sim.start_time)
                sim.start_time = datetime(2015,5,10,15,15,1)

        .. code-block::

            >>> datetime.datetime(2015,5,10,15,15,1)
        """
        return self._model.getSimulationDateTime(
            SimulationTime.StartDateTime.value)

    @start_time.setter
    def start_time(self, dtimeval):
        """Set simulation Start time"""
        self._model.setSimulationDateTime(SimulationTime.StartDateTime.value,
                                          dtimeval)

    @property
    def end_time(self):
        """Get/set Simulation end time.

        Examples:

        .. code-block:: python

            from pyswmm import Simulation

            with Simulation('tests/data/model_weir_setting.inp') as sim:
                print(sim.end_time)
                sim.end_time = datetime(2016,5,10,15,15,1)

        .. code-block::

            >>> datetime.datetime(2016,5,10,15,15,1)
        """
        return self._model.getSimulationDateTime(
            SimulationTime.EndDateTime.value)

    @end_time.setter
    def end_time(self, dtimeval):
        """Set simulation End time."""
        self._model.setSimulationDateTime(SimulationTime.EndDateTime.value,
                                          dtimeval)

    @property
    def report_start(self):
        """Get/set Simulation report start time.

        Examples:

        .. code-block:: python

            from pyswmm import Simulation

            with Simulation('tests/data/model_weir_setting.inp') as sim:
                print(sim.report_start)
                sim.report_start = datetime(2015,5,10,15,15,1)

        .. code-block::

            >>> datetime.datetime(2015,5,10,15,15,1)
        """
        return self._model.getSimulationDateTime(
            SimulationTime.ReportStart.value)

    @report_start.setter
    def report_start(self, dtimeval):
        """Set simulation report start time."""
        self._model.setSimulationDateTime(SimulationTime.ReportStart.value,
                                          dtimeval)

    @property
    def flow_units(self):
        """
        Get Simulation Units (CFS, GPM, MGD, CMS, LPS, MLD).

        :return: Flow Unit
        :rtype: str

        Examples:

        .. code-block:: python

            from pyswmm import Simulation

            with Simulation('tests/data/model_weir_setting.inp') as sim:
                print(sim.flow_units)

        .. code-block::

            >>> CFS
        """
        return self._model.getSimUnit(SimulationUnits.FlowUnits.value)

    @property
    def system_units(self):
        """Get system units (US, SI).

        :return: System Unit
        :rtype: str

        Examples:

        .. code-block:: python

            from pyswmm import Simulation

            with Simulation('tests/data/model_weir_setting.inp') as sim:
                print(sim.system_units)

        .. code-block::

            >>> US
        """
        return self._model.getSimUnit(SimulationUnits.UnitSystem.value)

    @property
    def current_time(self):
        """Get Simulation Current Time.

        :return: Current Simulation Time
        :rtype: Datetime

        Examples:

        .. code-block:: python

            from pyswmm import Simulation

            with Simulation('tests/data/model_weir_setting.inp') as sim:
                for step in sim:
                    print(sim.current_time)

        .. code-block::

            >>> 2015-11-01 14:00:30
            >>> 2015-11-01 14:01:00
            >>> 2015-11-01 14:01:30
            >>> 2015-11-01 14:02:00
        """
        return self._model.getCurrentSimulationTime()

    @property
    def percent_complete(self):
        """Get Simulation Percent Complete.

        :return: Current Percent Complete
        :rtype: Datetime

        Examples:

        .. code-block:: python

            from pyswmm import Simulation

            with Simulation('tests/data/model_weir_setting.inp') as sim:
                for step in sim:

        .. code-block::

            >>> 0.01
            >>> 0.25
            >>> 0.50
            >>> 0.75
        """
        dt = self.current_time - self.start_time
        total_time = self.end_time - self.start_time
        return float(dt.total_seconds()) / total_time.total_seconds()

    def use_hotstart(self,hotstart_file):
        """
        Use a hotstart file to initialize the simulation.

        This must be run before the simualation loop but within
        the simulation context manager.

        :param str hotstart_file: Path to hotstart file.

        .. code-block:: python

            with Simulation('model_weir_setting.inp') as sim:
                sim.use_hotstart("path_to_hotstart.hsf")

                for ind, step in enumerate(sim):
                    break

        """
        self._model.swmm_use_hotstart(hotstart_file)

    def save_hotstart(self,hotstart_file):

        """
        Save the current state of the model to a hotstart file.

        This can be run at any point during the simultion.

        :param str hotstart_file: Path to hotstart file.

        .. code-block:: python

            with Simulation('model_weir_setting.inp') as sim:
                for ind, step in enumerate(sim):
                    if ind == 10:
                        sim.save_hotstart('new_hsf.HSF')

        """
        self._model.swmm_save_hotstart(hotstart_file)

    def _before_start(self):
        """Get Before Start Callback.

        :return: Callbacks
        """
        return self._callbacks["before_start"]

    def add_before_start(self, callback):
        """
        Add callback function/method/object to execute before
        the simlation starts. Needs to be callable.

        :param func callback: Callable Object

        (See self.add_after_close() for more details)
        """
        if self._is_callback(callback):
            self._callbacks["before_start"] = callback

    def _after_start(self):
        """Get After Start Callback.

        :return: Callbacks
        """
        return self._callbacks["after_start"]

    def add_after_start(self, callback):
        """
        Add callback function/method/object to execute after
        a simlation start. Needs to be callable.  This callback allows
        setting initial link target_settings (such as an orifice).

        :param func callback: Callable Object

        (See self.add_after_close() for more details)
        """
        if self._is_callback(callback):
            self._callbacks["after_start"] = callback

    def _before_step(self):
        """Get Before Step Callback.

        :return: Callbacks
        """
        return self._callbacks["before_step"]

    def add_before_step(self, callback):
        """
        Add callback function/method/object to execute before
        a simlation step. Needs to be callable.

        :param func callback: Callable Object

        (See self.add_after_close() for more details)
        """
        if self._is_callback(callback):
            self._callbacks["before_step"] = callback

    def _after_step(self):
        """Get After Step Callback.

        :return: Callbacks
        """
        return self._callbacks["after_step"]

    def add_after_step(self, callback):
        """
        Add callback function/method/object to execute after
        a simlation step. Needs to be callable.

        :param func callback: Callable Object

        (See self.add_after_close() for more details)
        """
        if self._is_callback(callback):
            self._callbacks["after_step"] = callback

    def _before_end(self):
        """Get Before End Callback.

        :return: Callbacks
        """
        return self._callbacks["before_end"]

    def add_before_end(self, callback):
        """
        Add callback function/method/object to execute after
        the simulation ends. Needs to be callable.

        :param func callback: Callable Object

        (See self.add_after_close() for more details)
        """
        if self._is_callback(callback):
            self._callbacks["before_end"] = callback

    def _after_end(self):
        """Get After End Callback.

        :return: Callbacks
        """
        return self._callbacks["after_end"]

    def add_after_end(self, callback):
        """
        Add callback function/method/object to execute after
        the simulation ends. Needs to be callable.

        :param func callback: Callable Object

        (See self.add_after_close() for more details)
        """
        if self._is_callback(callback):
            self._callbacks["after_end"] = callback

    def _after_close(self):
        """Get After Close Callback.

        :return: Callbacks
        """
        return self._callbacks["after_close"]

    def add_after_close(self, callback):
        """
        Add callback function/method/object to execute after
        the simulation is closed. Needs to be callable.

        :param func callback: Callable Object

        .. code-block:: python

            from pyswmm import Simulation

            def test_callback():
                print("CALLBACK - Executed")

            with Simulation('tests/data/model_weir_setting.inp') as sim:

                sim.before_start(test_callback) #<- pass function handle.
                print("Waiting to Start")
                for ind, step in enumerate(sim):
                    print("Step {}".format(ind))
                print("Complete!")
            print("Closed")

        .. code-block::

            >>> "Waiting to Start"
            >>> "CALLBACK - Executed"
            >>> "Step 0"
            >>> "Step 1"
            >>> "Complete!"
            >>> "Closed"
        """
        if self._is_callback(callback):
            self._callbacks["after_close"] = callback
