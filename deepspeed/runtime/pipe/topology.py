# Copyright 2019 The Microsoft DeepSpeed Team
import os
import datetime
import itertools

from deepspeed import comm as dist

from collections import namedtuple
from itertools import product as cartesian_product


class ProcessTopology:
    """ Manages the mapping of n-dimensional Cartesian coordinates to linear
    indices. This mapping is used to map the rank of processes to the grid
    for various forms of parallelism.

    Each axis of the tensor is accessed by its name. The provided ordering
    of the axes defines the layout of the topology. ProcessTopology uses a "row-major"
    layout of the tensor axes, and so axes=['x', 'y'] would map coordinates (x,y) and
    (x,y+1) to adjacent linear indices. If instead axes=['y', 'x'] was used, coordinates
    (x,y) and (x+1,y) would be adjacent.

    Some methods return ProcessCoord namedtuples.
    """
    def __init__(self, axes, dims, custom_mapping=None):
        """Create a mapping of n-dimensional tensor coordinates to linear indices.

        Arguments:
            axes (list): the names of the tensor axes
            dims (list): the dimension (length) of each axis of the topology tensor
        """

        self.axes = axes  # names of each topology axis
        self.dims = dims  # length of each topology axis

        # This is actually a class that lets us hash {'row':3, 'col':2} mappings
        self.ProcessCoord = namedtuple('ProcessCoord', axes)

        self.mapping = {}
        if custom_mapping == None:
            ranges = [range(d) for d in dims]
            # example: 1, (0,0,1)
            for global_rank, coord in enumerate(cartesian_product(*ranges)):
                key = {axis: coord[self.axes.index(axis)] for axis in self.axes}
                key = self.ProcessCoord(**key)
                # for example, {ProcessCoord(row=0, col=1) : 1}
                self.mapping[key] = global_rank
        else:
            for rank_info in custom_mapping:
                rank = rank_info.rank
                if len(rank_info.active_coordinates) == 0:
                    continue
                coords = rank_info.active_coordinates[0]
                dp_id = coords[0]
                stage_id = coords[1]

                key = self.ProcessCoord(data=dp_id, pipe=stage_id)
                self.mapping[key] = rank

    def modify_mapping(self, rank, **axes):
        find = False
        for coord, idx in self.mapping.items():
            if idx == rank:
                find = True
                break
        if find:
            self.mapping[coord] = -1
        else:
            print(f'{self.mapping} not find {rank}, {type(rank)}')
        key = {axis: axes.get(axis) for axis in self.axes}
        key = self.ProcessCoord(**key)
        self.mapping[key] = rank

    def get_rank(self, **coord_kwargs):
        """Return the global rank of a process via its coordinates.

        Coordinates are specified as kwargs. For example:

            >>> X = ProcessTopology(axes=['x', 'y'], dims=[2,3])
            >>> X.get_rank(x=0, y=1)
            1
        """
        if len(coord_kwargs) != len(self.axes):
            raise ValueError('get_rank() does not support slices. Use filter_match())')

        key = self.ProcessCoord(**coord_kwargs)
        assert key in self.mapping, f'key {coord_kwargs} invalid'
        return self.mapping[key]

    def get_axis_names(self):
        """Return a list of the axis names in the ordering of the topology. """
        return self.axes

    def get_rank_repr(self,
                      rank,
                      omit_axes=['data',
                                 'pipe'],
                      inner_sep='_',
                      outer_sep='-'):
        """Return a string representation of a rank.

        This method is primarily used for checkpointing model data.

        For example:
            >>> topo = Topo(axes=['a', 'b'], dims=[2, 2])
            >>> topo.get_rank_repr(rank=3)
            'a_01-b_01'
            >>> topo.get_rank_repr(rank=3, omit_axes=['a'])
            'b_01'

        Args:
            rank (int): A rank in the topology.
            omit_axes (list, optional): Axes that should not be in the representation. Defaults to ['data', 'pipe'].
            inner_sep (str, optional): [description]. Defaults to '_'.
            outer_sep (str, optional): [description]. Defaults to '-'.

        Returns:
            str: A string representation of the coordinate owned by ``rank``.
        """
        omit_axes = frozenset(omit_axes)
        axes = [a for a in self.get_axis_names() if a not in omit_axes]
        names = []
        for ax in axes:
            ax_rank = getattr(self.get_coord(rank=rank), ax)
            names.append(f'{ax}{inner_sep}{ax_rank:02d}')
        return outer_sep.join(names)

    def get_dim(self, axis):
        """Return the number of processes along the given axis.

        For example:
            >>> X = ProcessTopology(axes=['x', 'y'], dims=[2,3])
            >>> X.get_dim('y')
            3
        """
        if axis not in self.axes:
            return 0
        return self.dims[self.axes.index(axis)]

    def get_coord(self, rank):
        """Return the coordinate owned by a process rank.

        The axes of the returned namedtuple can be directly accessed as members. For
        example:
            >>> X = ProcessTopology(axes=['x', 'y'], dims=[2,3])
            >>> coord = X.get_coord(rank=1)
            >>> coord.x
            0
            >>> coord.y
            1
        """
        for coord, idx in self.mapping.items():
            if idx == rank:
                return coord
        raise ValueError(f'rank {rank} not found in topology.')

    def get_axis_comm_lists(self, axis):
        """ Construct lists suitable for a communicator group along axis ``axis``.

        Example:
            >>> topo = Topo(axes=['pipe', 'data', 'model'], dims=[2, 2, 2])
            >>> topo.get_axis_comm_lists('pipe')
            [
                [0, 4], # data=0, model=0
                [1, 5], # data=0, model=1
                [2, 6], # data=1, model=0
                [3, 7], # data=1, model=1
            ]

        Returns:
            A list of lists whose coordinates match in all axes *except* ``axis``.
        """

        # We don't want to RuntimeError because it allows us to write more generalized
        # code for hybrid parallelisms.
        if axis not in self.axes:
            return []

        # Grab all axes but `axis`
        other_axes = [a for a in self.axes if a != axis]

        lists = []

        # Construct all combinations of coords with other_axes
        ranges = [range(self.get_dim(a)) for a in other_axes]
        for coord in cartesian_product(*ranges):
            other_keys = {a: coord[other_axes.index(a)] for a in other_axes}
            # now go over all ranks in `axis`.
            sub_list = []
            for axis_key in range(self.get_dim(axis)):
                key = self.ProcessCoord(**other_keys, **{axis: axis_key})
                sub_list.append(self.mapping[key])
            lists.append(sub_list)

        return lists

    def filter_match(self, **filter_kwargs):
        """Return the list of ranks whose coordinates match the provided criteria.

        Example:
            >>> X = ProcessTopology(axes=['pipe', 'data', 'model'], dims=[2, 2, 2])
            >>> X.filter_match(pipe=0, data=1)
            [2, 3]
            >>> [X.get_coord(rank) for rank in X.filter_match(pipe=0, data=1)]
            [ProcessCoord(pipe=0, data=1, model=0), ProcessCoord(pipe=0, data=1, model=1)]

        Arguments:
            **filter_kwargs (dict): criteria used to select coordinates.

        Returns:
            The list of ranks whose coordinates match filter_kwargs.
        """
        def _filter_helper(x):
            for key, val in filter_kwargs.items():
                if getattr(x, key) != val:
                    return False
            return True

        coords = filter(_filter_helper, self.mapping.keys())
        return [self.mapping[coord] for coord in coords]

    def get_axis_list(self, axis, idx):
        """Returns the list of global ranks whose coordinate in an axis is idx.

        For example:
            >>> X = ProcessTopology(axes=['x', 'y'], dims=[2,3])
            >>> X.get_axis_list(axis='x', idx=0)
            [0, 1, 2]
            >>> X.get_axis_list(axis='y', idx=0)
            [0, 3]
        """

        # This could be faster by generating the desired keys directly instead of
        # filtering.
        axis_num = self.axes.index(axis)
        ranks = [self.mapping[k] for k in self.mapping.keys() if k[axis_num] == idx]
        return ranks

    def world_size(self):
        return len(self.mapping)

    def __str__(self):
        return str(self.mapping)


def _prime_factors(N):
    """ Returns the prime factorization of positive integer N. """
    if N <= 0:
        raise ValueError("Values must be strictly positive.")

    primes = []
    while N != 1:
        for candidate in range(2, N + 1):
            if N % candidate == 0:
                primes.append(candidate)
                N //= candidate
                break
    return primes


class PipeDataParallelTopology(ProcessTopology):
    """ A topology specialization for hybrid data and pipeline parallelism.

        Uses data parallelism on the last dimension to encourage gradient
        reductions to use high-bandwidth intra-node links and lower-volume
        pipeline communications to use low-bandwidth inter-node links.
    """
    def __init__(self, num_pp, num_dp, custom_mapping=None):
        super().__init__(axes=['pipe', 'data'], dims=[num_pp, num_dp], custom_mapping=custom_mapping)


class PipeModelDataParallelTopology(ProcessTopology):
    """ A topology for hybrid pipeline, model, and data parallelism. """
    def __init__(self, num_pp, num_mp, num_dp):
        super().__init__(axes=['pipe', 'data', 'model'], dims=[num_pp, num_dp, num_mp])


class PipelineParallelGrid:
    """Implements a grid object that stores the data parallel ranks
    corresponding to each of the model parallel stages

    The grid object organizes the processes in a distributed pytorch job
    into a 2D grid, of stage_id and data_parallel_id.

    self.stage_id and self.data_parallel_id stores the stage id
    and the data parallel id of current process.

    self.dp_group groups the processes by stage_id.
    self.dp_group[i], is a list containing all process ranks whose
    stage_id is i.

    self.p2p_groups stores a list of tuple, where each tuple
    stores process ranks of adjacent stages for a given data_parallel_id.
    For example if num_stage is 5 then a tuple [7,8] represents stages [3, 4],
    with data_parallel id = 1. A stage wrap around will appear as non-adjacent ranks,
    for example tuple [4,0] with representing wrap-around stage 4 and 0, for
    data_parallel_id = 0, or similarly [9,5] represents wrapped around stages [4,0]
    for data_parallel_id = 1.
    """
    def __init__(self, topology=None, process_group=None, candidate_pool=None):
        # TODO use process_group if provided
        self.global_rank = dist.get_rank()
        self.local_rank = dist.get_local_rank()
        self.local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
        self.world_size = dist.get_world_size()
        if topology is not None:
            if self.global_rank == 0:
                print('Using topology:', topology)
            self._topo = topology
        else:
            num_pp = 1
            num_dp = 1
            for idx, prime in enumerate(_prime_factors(self.world_size)):
                if idx % 2 == 0:
                    num_pp *= prime
                else:
                    num_dp *= prime
            self._topo = PipeDataParallelTopology(num_dp=num_dp, num_pp=num_pp)
        self.data_parallel_size = max(self._topo.get_dim('data'), 1)
        self.pipe_parallel_size = max(self._topo.get_dim('pipe'), 1)
        self.model_parallel_size = max(self._topo.get_dim('model'), 1)
        self.slice_parallel_size = self.model_parallel_size

        # local group
        if self.local_world_size > 1:
            for gr in range(self.world_size // self.local_world_size):
                ranks = [i + gr * self.local_world_size for i in range(self.local_world_size)]
                timeout = self.approximate_timeout_value(ranks, type='dp_group')
                group = dist.new_group(ranks, timeout=timeout)
                if self.global_rank in ranks:
                    self.local_proc_group = group

        # Adaptive communication module
        self.active_ranks = list(range(self.world_size))
        self.waiting_rank_coordinates = {}
        if candidate_pool is not None:
            for rank in candidate_pool:
                self.set_waiting(rank, candidate_pool[rank])

        assert self._is_grid_valid(), "Invalid Grid"

        self.stage_id = self.get_stage_id()
        self.data_parallel_id = self.get_data_parallel_id()

        # Create a Gloo group
        self.gloo_pg = dist.new_group(list(range(self.world_size)), backend='gloo')
        self.cpu_transfer = os.environ.get('CPU_TRANSFER', '') == 'ON'

        # Create new ProcessGroups for all model parallelism. DeepSpeedLight uses these
        # to detect overflow, etc.
        self.ds_model_proc_group = None
        self.ds_model_rank = -1
        for dp in range(self.data_parallel_size):
            ranks = sorted(self._topo.get_axis_list(axis='data', idx=dp))
            if self.global_rank == 0:
                #print(f'RANK={self.global_rank} building DeepSpeed model group: {ranks}')
                pass
            proc_group = dist.new_group(ranks=ranks)
            if self.global_rank in ranks:
                self.ds_model_proc_group = proc_group
                self.ds_model_world_size = len(ranks)
                self.ds_model_rank = ranks.index(self.global_rank)
        if not self.is_active:
            self.ds_model_proc_group = self.waiting_group
            self.ds_model_world_size = 1
            self.ds_model_rank = 0
        assert self.ds_model_rank > -1
        assert self.ds_model_proc_group is not None

        # Create new ProcessGroup for gradient all-reduces - these are the data parallel groups
        self.dp_group = []
        self.dp_groups = self._topo.get_axis_comm_lists('data')
        for g in self.dp_groups:
            timeout = self.approximate_timeout_value(g, type='dp_group')
            proc_group = dist.new_group(ranks=g, timeout=timeout)
            if self.global_rank in g:
                self.dp_group = g
                self.dp_proc_group = proc_group
            if self.cpu_transfer:
                proc_group = dist.new_group(ranks=g, timeout=timeout, backend='gloo')
                if self.global_rank in g:
                    self.dp_proc_cpu_group = proc_group
        if not self.is_active:
            self.dp_group = [rank]
            self.dp_proc_group = self.waiting_group

        self.is_first_stage = (self.stage_id == 0)
        self.is_last_stage = (self.stage_id == (self.pipe_parallel_size - 1))

        self.p2p_groups = self._build_p2p_groups()

        # Create new ProcessGroup for pipeline collectives - these are pipe parallel groups
        self.pp_group = []
        self.pp_proc_group = None
        self.pipe_groups = self._topo.get_axis_comm_lists('pipe')
        for ranks in self.pipe_groups:
            if self.global_rank == 0:
                #print(f'RANK={self.global_rank} building pipeline group: {ranks}')
                pass
            proc_group = dist.new_group(ranks=ranks)
            if self.global_rank in ranks:
                self.pp_group = ranks
                self.pp_proc_group = proc_group
        if not self.is_active:
            self.pp_group = [rank]
            self.pp_proc_group = self.waiting_group
        assert self.pp_proc_group is not None

        # Adaptive communication module
        # self.active_ranks = list(range(self.world_size))
        # self.waiting_rank_coordinates = {}
        # self.waiting_ranks = []
        # self.normal_pipelines = set(range(self.data_parallel_size))

        # self.stage_active_replicas = {}
        # self.stage_waiting_replicas = {}
        # for stage_id, ranks in enumerate(self._topo.get_axis_comm_lists('data')):
        #     self.stage_active_replicas[stage_id] = ranks
        #     self.stage_waiting_replicas[stage_id] = []

        # timeout = self.approximate_timeout_value(self.active_ranks, type='active_group')
        # self.init_data_parallel_fallback_group()

        # Create new ProcessGroup for model (tensor-slicing) collectives

        # Short circuit case without model parallelism.
        # TODO: it would be nice if topology had bcast semantics to avoid this branching
        # case?
        if self.model_parallel_size == 1:
            for group_rank in range(self.world_size):
                group_rank = [group_rank]
                group = dist.new_group(ranks=group_rank)
                if group_rank[0] == self.global_rank:
                    self.slice_group = group_rank
                    self.slice_proc_group = group
            return
        else:
            self.mp_group = []
            self.model_groups = self._topo.get_axis_comm_lists('model')
            for g in self.model_groups:
                proc_group = dist.new_group(ranks=g)
                if self.global_rank in g:
                    self.slice_group = g
                    self.slice_proc_group = proc_group

    def get_coord(self, rank):
        if rank in self.get_active_ranks():
            return self._topo.get_coord(rank)
        return self.get_waiting_rank_coordinate(rank)

    def get_stage_id(self):
        return self.get_coord(rank=self.global_rank).pipe

    def get_data_parallel_id(self):
        return self.get_coord(rank=self.global_rank).data

    def _build_p2p_groups(self):
        """Groups for sending and receiving activations and gradients across model
        parallel stages.
        """
        comm_lists = self._topo.get_axis_comm_lists('pipe')
        p2p_lists = []
        for rank in range(self.world_size):
            for l in comm_lists:
                assert len(l) == self.pipe_parallel_size
                if rank in l:
                    idx = l.index(rank)
                    buddy_rank = l[(idx + 1) % self.pipe_parallel_size]
                    p2p_lists.append([rank, buddy_rank])
                    break  # next global rank
        assert len(p2p_lists) == len(self.get_active_ranks())
        return p2p_lists

    def approximate_timeout_value(self, ranks, type):
        if not dist.spotdl_enabled:
            return None
        # type in ['dp_group', 'p2p_group', 'mp_group']
        return datetime.timedelta(seconds=60)

    def init_data_parallel_fallback_group(self):
        self.dp_fallback_groups = {}
        # default dp group
        # key = tuple(sorted(self.dp_group))
        # self.dp_fallback_groups[key] = self.dp_proc_group

        for dp in range(self.data_parallel_size):
            ranks = sorted(self._topo.get_axis_list(axis='data', idx=dp))

            for l in range(1, len(ranks) + 1):
                for subset in itertools.combinations(ranks, l):
                    timeout = self.approximate_timeout_value(subset, type='dp_group')
                    group = dist.new_group(ranks=subset, timeout=timeout)
                    if self.global_rank in subset:
                        key = tuple(sorted(subset))
                        self.dp_fallback_groups[key] = group

    def _is_grid_valid(self):
        ranks = 1
        for ax in self._topo.get_axis_names():
            ranks *= self._topo.get_dim(ax)
        return ranks == len(self.get_active_ranks())

    #returns the global rank of the process with the provided stage id
    #which has the same data_parallel_id as caller process
    def stage_to_global(self, stage_id, **kwargs):
        me = self.get_coord(self.global_rank)
        transform = me._replace(pipe=stage_id, **kwargs)._asdict()
        return self._topo.get_rank(**transform)

    @property
    def is_active(self):
        return self.global_rank in self.active_ranks

    @property
    def num_waiting(self):
        return len(self.waiting_rank_coordinates)

    def get_active_ranks(self):
        return self.active_ranks

    def get_waiting_ranks(self):
        return self.waiting_ranks

    def get_live_ranks(self):
        return self.active_ranks + self.waiting_ranks

    def get_normal_pipelines(self):
        return self.normal_pipelines

    def get_stage_active_replicas(self):
        return self.stage_active_replicas

    def get_stage_waiting_replicas(self):
        return self.stage_waiting_replicas

    def intra_migration_update(self, normal_pipelines, stage_active_replicas, stage_waiting_replicas, lost_nodes):
        self.active_ranks, self.waiting_ranks = [], []
        for stage_id, ranks in stage_active_replicas.items():
            self.stage_active_replicas[int(stage_id)] = ranks
            self.active_ranks.extend(ranks)

            if self.global_rank in ranks:
                self.dp_group = ranks
                key = tuple(sorted(ranks))
                self.old = self.dp_proc_group
                self.dp_proc_group = self.dp_fallback_groups[key]

        for stage_id, ranks in stage_waiting_replicas.items():
            self.stage_waiting_replicas[int(stage_id)] = ranks
            self.waiting_ranks.extend(ranks)
        self.normal_pipelines = set(normal_pipelines)

    def remove_node(self, rank):
        pass

    def set_waiting(self, rank, coordinate):
        self.active_ranks.remove(rank)
        self.waiting_rank_coordinates[rank] = self._topo.ProcessCoord(
            data=coordinate[0], pipe=coordinate[1]
        )
        waiting_group = dist.new_group(ranks=[rank])
        if rank == self.global_rank:
            self.waiting_group = waiting_group

    def get_waiting_rank_coordinate(self, rank):
        return self.waiting_rank_coordinates[rank]

    def topology(self):
        return self._topo

    # MPU functions for DeepSpeed integration
    def get_global_rank(self):
        return self.global_rank

    def get_pipe_parallel_rank(self):
        """ The stage of the pipeline this rank resides in. """
        return self.get_stage_id()

    def get_pipe_parallel_world_size(self):
        """ The number of stages in the pipeline. """
        return self.pipe_parallel_size

    def get_pipe_parallel_group(self):
        """ The group of ranks within the same pipeline. """
        return self.pp_proc_group

    def get_data_parallel_rank(self):
        """ Which pipeline this rank resides in. """
        return self.data_parallel_id

    def get_data_parallel_world_size(self):
        """ The number of pipelines. """
        return self.data_parallel_size

    def get_data_parallel_group(self):
        """ The group of ranks within the same stage of all pipelines. """
        return self.dp_proc_group

    # These are model parallel groups across all types of model parallelism.
    # Deepspeed uses them to detect overflow, etc.
    def get_model_parallel_rank(self):
        return self.ds_model_rank

    def get_model_parallel_world_size(self):
        return self.ds_model_world_size

    def get_model_parallel_group(self):
        return self.ds_model_proc_group

    # For Megatron-style tensor slicing
    def get_slice_parallel_rank(self):
        if 'model' in self._topo.get_axis_names():
            return self.get_coord(rank=self.global_rank).model
        else:
            return 0

    def get_slice_parallel_world_size(self):
        return self.slice_parallel_size

    def get_slice_parallel_group(self):
        return self.slice_proc_group
