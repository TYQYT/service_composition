#python3
# coding=utf-8
import random
import matplotlib.pyplot as plt
import numpy as np
import math
import json
import time

class CssProblemModel:
    '''CssProblemModel - the problem model of CSS

    Attributes:
        css : Cloud Service Set, a list of ms(Menufeature Service)
                ms is a list of services 
                the structure of css data is :
                [ [s_0_0] , [s_0_1] , ... , [s_0_m] ],
                [ [s_1_0] , [s_1_1] , ... , [s_1_m] ],
                [   ...   ,   ...   , ... ,   ...   ],
                [ [s_n_0] , [s_n_1] , ... , [s_n_m] ],
    '''

    def __init__(self, file_path: str):
        '''Default init - init CSS with random data
        '''
        self.css = []
        if file_path is not None:
            self._init_with_file_data(file_path)

    def _init_with_file_data(self, file_path: str):
        with open(file_path, 'r') as f:
            data = json.load(f)
            ms_lst = data['css_set']
            print('ms number : {:d}'.format(len(ms_lst)))
            for i, s_lst in enumerate(ms_lst):
                print('\t{:d}. s number : {:d}'.format(i, len(s_lst)))
                ms = []
                for s_d in s_lst:
                    ms.append(CssProblemModel.Service.from_dict(s_d))
                self.css.append(ms)

            self.link_set = data['css_link_set']

    def _get_index(self, ms: list):
        '''_get_index - Get a unique index of ms.
        ** If need, over write me.
        '''
        can_visit = []
        for i, s in enumerate(ms):
            if not s.visited:
                can_visit.append(i)
        idx = can_visit[random.randint(0, len(can_visit)-1)]
        ms[idx].visited = 1
        return idx

    def _get_index_by_old_idx(self, ms: list, idx1: int, idx2: int, search_ops):
        '''_get_index - Get a unique index of ms, independent idx1 and idx2
        ** If need, over write me.
        '''
        idx = search_ops(idx1, idx2, len(ms))

        if not ms[idx].visited:
            ms[idx].visited = 1
            return idx

        can_visit = []
        for i, s in enumerate(ms):
            if not s.visited:
                can_visit.append(i)
        idx = can_visit[random.randint(0, len(can_visit)-1)]
        ms[idx].visited = 1
        return idx

    def _clear_visited(self):
        '''_clear_visited - Clear visited flag
        '''
        for ms in self.css:
            for s in ms:
                s.visited = 0

    def get_new_path(self, path_num: int = 1):
        '''Get random path_num paths restore to path_lst
        '''
        self._clear_visited()

        res = []
        for _ in range(path_num):
            one_path = []
            for ms in self.css:
                i = self._get_index(ms)
                one_path.append(i)
            res.append(one_path)

        return res

    def get_new_path_by_probabilty(self, path_lst: list, p: float):
        self._clear_visited()
        ret_path_lst = path_lst
        for ret_path in ret_path_lst:
            for ms_idx, ms in enumerate(self.css):
                if random.random() < p:
                    ret_path[ms_idx] = self._get_index(ms)
        return ret_path_lst

    def get_new_path_by_old_path(self, path_lst1: list, path_lst2: list,
            search_ops):
        '''Get new path by path1 and path2
        '''
        if len(path_lst1) != len(path_lst2):
            print('Err path_lst2 != path_lst1')
            return None

        self._clear_visited()

        ret_path_lst = path_lst1
        ms_idx = random.randint(0, len(self.css)-1)
        for ret_path, path1, path2 in zip(ret_path_lst, path_lst1, path_lst2):
            idx = self._get_index_by_old_idx(
                    self.css[ms_idx], path1[ms_idx], path2[ms_idx], search_ops)
            ret_path[ms_idx] = idx

        return ret_path_lst

    def cal_qos(self, path_lst: list):
        '''cal_qos - calculate qos
        '''
        total_service = CssProblemModel.Service()
        link_val = 0
        for path in path_lst:
            one_path_service = CssProblemModel.Service()
            for idx, ms in zip(path, self.css):
                if idx == 2 or idx == 3:
                    one_path_service.parallel_add(ms[idx])
                else:
                    one_path_service.sequence_add(ms[idx])
            total_service.parallel_add(one_path_service)

            # 0 -> 1 -> 2 -> 3 -> 4
            # 0 -> 1 -> 2 -> 4
            #           3

            for i, i0, i1 in [
                (0, path[0], path[1]),
                (1, path[1], path[2]),
                (2, path[1], path[3]),
                (2, path[2], path[4]),
                (3, path[3], path[4]),
            ]:
                link_val += self.link_set[i][i0][i1]

            #for i, i0, i1 in zip(range(len(path)), path, path[1:]):
            #    link_val += self.link_set[i][i0][i1]

        return (total_service.cal_qos() + link_val) / 2

    class Service:
        '''Service - class of service

        Static Attributes:
            W_T  : the weight of time
            W_P  : the weight of price
            W_RE : the weight of reliability
            W_AV : the weight of availability

        Attributes:
            t  : Time spent to the service
            p  : The price of the service
            re : Service reliability
            av : Service availability
        '''

        W_T  = 0.25
        W_P  = 0.25
        W_RE = 0.25
        W_AV = 0.25

        MIN_T  = 1.0
        MIN_P  = 1.0
        MIN_RE = 0.75
        MIN_AV = 0.75

        MAX_T  = 20.0
        MAX_P  = 100.0
        MAX_RE = 1.0
        MAX_AV = 1.0

        def __init__(self,
                time          : float = MAX_T,
                price         : float = MAX_P,
                reliability   : float = MIN_RE,
                availability  : float = MIN_AV,
                normalization : bool  = True) :
            if normalization:
                self.t  = self._normalization(time,
                        self.MIN_T, self.MAX_T, True)
                self.p  = self._normalization(price,
                        self.MIN_P, self.MAX_P, True)
                self.re = self._normalization(reliability,
                        self.MIN_RE, self.MAX_RE, False)
                self.av = self._normalization(availability,
                        self.MIN_AV, self.MAX_AV, False)
            else:
                self.t  = time
                self.p  = price
                self.re = reliability
                self.av = availability
            self.visited = 0

        def __str__(self):
            return '-> {:0>2.2f}, {:0>2.2f}, {:0>2.2f}, {:0>2.2f}'\
                .format(self.t, self.p, self.re, self.av)

        @classmethod
        def to_dict(cls, s):
            return {
                'time': s.t,
                'price': s.p,
                'reliability': s.re,
                'availability': s.av
            }

        @classmethod
        def from_dict(cls, d: dict):
            return cls(d['time'], d['price'],
                    d['reliability'], d['availability'])

        def parallel_add(self, other):
            self.t  = max(self.t, other.t)
            self.p  += other.p
            self.re *= other.re
            self.av *= other.av

        def sequence_add(self, other):
            self.t  += other.t
            self.p  += other.p
            self.re *= other.re
            self.av *= other.av

        def _normalization(self, 
                v: float, 
                min_v: float, 
                max_v: float, 
                is_cost: bool):
            if is_cost:
                return (max_v - v) / (max_v - min_v)
            else:
                return (v - min_v) / (max_v - min_v)

        def cal_qos(self):
            '''cal_qos - Calculate QOS(Quality Of Service)
            '''
            return self.W_T * self.t + self.W_P * self.p + \
                self.W_RE * self.re + self.W_AV * self.av

class ABCAlgo:
    '''ABCAlgo - ABC algorithm model

    Attributes:
        problem           : CssProblemModel
        leader_num        : leader number
        follower_num      : followe number
        task_num          : Task number
        max_iteration_num : max iteration number
        resolution_src    : resolution source, a list of ResolutionSource
        best_resolution   : best resolution, a ResolutionSource
        result_record     : record the best_resolution of each iterations
    '''
    LIMIT = 100
    class ResolutionSource:
        '''ResolutionSource

        Attributes:
            path_lst : path list
            trail    : count of qos no change
        '''
        problem = None
        def __init__(self, path_lst: list):
            self.path_lst = path_lst
            self.fitness = self.problem.cal_qos(self.path_lst)
            self.trail = 0

        def __eq__(self, other) -> bool:
            return self.fitness == other.fitness

    def __init__(self, problem: CssProblemModel,
            leader_num: int,
            follower_num: int,
            max_iter_num: int,
            task_num: int):
        self.ResolutionSource.problem = problem
        self.problem = problem
        self.leader_num = leader_num
        self.follower_num = follower_num
        self.max_iteration_num = max_iter_num
        self.task_num = task_num
        self.resolution_src = []
        self.resolution_trail = []
        self.best_resolution = self.ResolutionSource(problem.get_new_path(task_num))
        self.result_record = []
        for _ in range(self.leader_num):
            self.resolution_src.append(self.ResolutionSource(self.problem.get_new_path(task_num)))

    def search_ops(self, idx1: int, idx2: int, lenght: int) -> int:
        return int(idx1 + random.random() * abs(idx1 - idx2)) % lenght

    def _update_resolution_src(self, index):
        src = self.resolution_src[index]
        src_another = self.ResolutionSource(self.problem.get_new_path(self.task_num))
        while src_another == src:
            src_another = self.ResolutionSource(self.problem.get_new_path(self.task_num))

        src_new = self.ResolutionSource(self.problem.get_new_path_by_old_path(src.path_lst, src_another.path_lst, self.search_ops))
        if src_new.fitness > src.fitness:
            self.resolution_src[index] = src_new
        else:
            self.resolution_src[index].trail += 1

    def _leader_procedure(self):
        for i in range(len(self.resolution_src)):
            self._update_resolution_src(i)

    def _follower_procedure(self):
        sum_fitness = 0
        for src in self.resolution_src:
            sum_fitness += src.fitness

        for i in range(len(self.resolution_src)):
            p = self.resolution_src[i].fitness / sum_fitness
            for _ in range(self.follower_num):
                if random.random() < p:
                    self._update_resolution_src(i)

    def _update_best_resolution(self):
        for src in self.resolution_src:
            if src.fitness > self.best_resolution.fitness:
                self.best_resolution = src

    def _scout_procedure(self):
        for i, src in enumerate(self.resolution_src):
            if src.trail > self.LIMIT:
                self.resolution_src[i] = self.ResolutionSource(self.problem.get_new_path(self.task_num))

    def slove(self):
        for _ in range(self.max_iteration_num):
            self._leader_procedure()
            self._follower_procedure()
            self._scout_procedure()
            self._update_best_resolution()
            self.result_record.append(self.best_resolution)
        #    finish_percent = (i+1) / self.max_iteration_num
        #    print('拼命计算中：{:.2f}% [{:d}/{:d}]\t'.format(
        #        i/self.max_iteration_num*100, i, self.max_iteration_num),
        #        '{:s}{:s}'.format(
        #            '#'*int(finish_percent*50), '-'*int(50-finish_percent*50)),
        #         end='\r')
        #print('\n***Finished***')

    def show_result(self):
        res_lst = []
        for res in self.result_record:
            print(res.path_lst, res.fitness)
            res_lst.append(res.fitness)

        plt.subplot(1, 2, 1)
        for path in self.best_resolution.path_lst:
            plt.plot(path, marker='o')
        plt.title('best resolution, fitness is {:.2f}'.format(
            self.best_resolution.fitness))

        plt.subplot(1, 2, 2)
        plt.plot(res_lst)#, marker='o')
        plt.title('result record')

        plt.tight_layout()
        plt.show()



import heapq as hq

class ABCAlgo_Elite(ABCAlgo):
    MR = 0.5
    ELITE_PER = 0.1

    def __init__(self, problem: CssProblemModel,
        leader_num: int,
        follower_num: int,
        max_iter_num: int,
        task_num: int):
        super().__init__(problem, leader_num, follower_num, max_iter_num, task_num)
        self.eliet_res_src = self._cal_eliet_resolution_src()

    def _cal_eliet_resolution_src(self):
        return hq.nlargest(int(ABCAlgo_Elite.ELITE_PER*len(self.resolution_src)),
            self.resolution_src, key=lambda res: res.fitness)

    def _update_resolution_src(self, index):
        if random.random() > ABCAlgo_Elite.MR:
            self.resolution_src[index].trail += 1
            return

        src = self.resolution_src[index]
        src_another = random.choice(self.eliet_res_src)

        src_new = self.ResolutionSource(self.problem.get_new_path_by_old_path(src.path_lst, src_another.path_lst, self.search_ops))
        if src_new.fitness > src.fitness:
            self.resolution_src[index] = src_new
        else:
            self.resolution_src[index].trail += 1

    def _update_best_resolution(self):
        self.eliet_res_src = self._cal_eliet_resolution_src()
        if self.eliet_res_src[0].fitness > self.best_resolution.fitness:
            self.best_resolution = self.eliet_res_src[0]

class ABCAlgo_Elite_2(ABCAlgo_Elite):
    def __init__(self, problem: CssProblemModel,
        leader_num: int,
        follower_num: int,
        max_iter_num: int,
        task_num: int,
        elite_iter_num: int):
        super().__init__(problem, leader_num, follower_num, max_iter_num, task_num)
        self.eliet_iter_num  = elite_iter_num
        self.eliet_iter = 0

    def _update_best_resolution(self):
        if self.eliet_iter == self.eliet_iter_num:
            self.eliet_iter = 0
            self.eliet_res_src = self._cal_eliet_resolution_src()
        if self.eliet_res_src[0].fitness > self.best_resolution.fitness:
            self.best_resolution = self.eliet_res_src[0]
        self.eliet_iter += 1

class ABCAlog_Adaptive(ABCAlgo):
    def __init__(self, problem: CssProblemModel,
        leader_num: int,
        follower_num: int,
        max_iter_num: int,
        task_num: int):
        super().__init__(problem, leader_num, follower_num, max_iter_num, task_num)

    def search_ops_for_leader(self, idx1: int, idx2: int, lenght: int) -> int:
        # 这里 fitness 先通过 self 传进来，但一定有更好的解决方案，后续在改
        if self.src_another_fitness > self.src_fitness:
            return int(idx1 + random.uniform(0.95, 1.5) * abs(idx1 - idx2) * \
                math.atan(self.src_another_fitness/self.src_fitness)) % lenght
        else:
            return int(idx1 - random.uniform(1.2, 1.6) * abs(idx1 - idx2) * \
                math.atan(self.src_fitness/self.src_another_fitness)
                + random.uniform(0.5, 1.5) * idx1 ) % lenght

    def search_ops_for_follower(self, idx1: int, idx2: int, lenght: int) -> int:
        if self.src_another_fitness > self.src_fitness:
            return int(idx2 + random.uniform(-0.45, 0.45) * abs(idx1 - idx2) * \
                math.atan(self.src_another_fitness/self.src_fitness)**(-1)) % lenght
        else:
            return int(idx1 - random.uniform(-0.45, 0.45) * abs(idx1 - idx2) * \
                math.atan(self.src_another_fitness/self.src_fitness)**(-1)) % lenght

    def _update_resolution_src_for_leader(self, index):
        src = self.resolution_src[index]
        src_another = random.choice(self.resolution_src)
        while src_another == src:
            src_another = random.choice(self.resolution_src)

        self.src_fitness = src.fitness
        self.src_another_fitness = src_another.fitness
        src_new = self.ResolutionSource(self.problem.get_new_path_by_old_path(src.path_lst, src_another.path_lst, self.search_ops_for_leader))
        if src_new.fitness > src.fitness:
            self.resolution_src[index] = src_new
        else:
            self.resolution_src[index].trail += 1
        pass

    def _update_resolution_src_for_follower(self, index):
        src = self.resolution_src[index]
        src_another = random.choice(self.resolution_src)
        while src_another == src:
            src_another = random.choice(self.resolution_src)

        self.src_fitness = src.fitness
        self.src_another_fitness = src_another.fitness
        src_new = self.ResolutionSource(self.problem.get_new_path_by_old_path(src.path_lst, src_another.path_lst, self.search_ops_for_follower))
        if src_new.fitness > src.fitness:
            self.resolution_src[index] = src_new
        else:
            self.resolution_src[index].trail += 1
        pass

    def _leader_procedure(self):
        for i in range(len(self.resolution_src)):
            self._update_resolution_src_for_leader(i)

    def _follower_procedure(self):
        sum_fitness = 0
        for src in self.resolution_src:
            sum_fitness += src.fitness

        for i in range(len(self.resolution_src)):
            p = self.resolution_src[i].fitness / sum_fitness
            for _ in range(self.follower_num):
                if random.random() < p:
                    self._update_resolution_src_for_follower(i)

class ABCAlgo_Islands:
    def __init__(self, problem: CssProblemModel,
        leader_num: int,
        follower_num: int,
        abc_iter_num: int,
        task_num: int,
        island_iter_num: int,
        island_num: int):
        self.island_iter_num = island_iter_num
        self.islands_lst = [
            ABCAlgo(problem, leader_num, follower_num, abc_iter_num, task_num) for _ in range(island_num)
        ]
        self.best_resolution = self.islands_lst[0].best_resolution
        
    def _cal_poor_resolution_src(self, island: ABCAlgo):
        return hq.nsmallest(2, island.resolution_src, key=lambda res: res.fitness)

    def _swap_island_person(self):
        swap_lst = []
        for island in self.islands_lst:
            island.resolution_src.sort(key=lambda res: res.fitness)
            swap_lst.append(island.resolution_src[-2:])
            del island.resolution_src[-2:]

        for i, res in enumerate(swap_lst):
            idx = (i + 1) % len(self.islands_lst)
            self.islands_lst[idx].resolution_src += res

    def slove(self):
        for _ in range(self.island_iter_num):
            for island in self.islands_lst:
                island.slove()
            self._swap_island_person()
        self.best_island = self.islands_lst[0]
        for island in self.islands_lst[1:]:
            if self.best_island.best_resolution.fitness < island.best_resolution.fitness:
                self.best_island = island
        self.best_resolution = self.best_island.best_resolution

    def show_result(self):
        self.best_island.show_result()

class ABCAlgo_Islands_Mixer(ABCAlgo_Islands):
    def __init__(self, problem: CssProblemModel,
        leader_num: int,
        follower_num: int,
        abc_iter_num: int,
        task_num: int,
        island_iter_num: int,
        island_num: int):
        self.island_iter_num = island_iter_num
        self.islands_lst = [
            ABCAlgo(problem, leader_num, follower_num, abc_iter_num, task_num) for _ in range(island_num)
        ] + [
            ABCAlgo_Elite(problem, leader_num, follower_num, abc_iter_num, task_num) for _ in range(island_num)
        ] + [
            ABCAlog_Adaptive(problem, leader_num, follower_num, abc_iter_num, task_num) for _ in range(island_num)
        ]
        self.best_resolution = self.islands_lst[0].best_resolution


def get_algo_class(algo_class, pm, iter_num: int):
    if algo_class == ABCAlgo:
        return ABCAlgo(pm, 100, 100, iter_num, 1)
    elif algo_class == ABCAlgo_Elite:
        return ABCAlgo_Elite(pm, 100, 100, iter_num, 1)
    elif algo_class == ABCAlgo_Elite_2:
        return ABCAlgo_Elite_2(pm, 100, 100, iter_num, 1, 5)
    elif algo_class == ABCAlog_Adaptive:
        return ABCAlog_Adaptive(pm, 100, 100, iter_num, 1)
    elif algo_class == ABCAlgo_Islands:
        return ABCAlgo_Islands(pm, 10, 10, 5, 1, int(iter_num / 5), 5)
    elif algo_class == ABCAlgo_Islands_Mixer:
        return ABCAlgo_Islands_Mixer(pm, 10, 10, 5, 1, int(iter_num / 5), 5)
    else:
        print('Not support')

if __name__ == "__main__":
    pm = CssProblemModel('./ABC_data_7.json')
    algo = ABCAlgo(pm, 50, 50, 100, 1)
    #algo = ABCAlgo_Elite(pm, 100, 100, 100, 1)
    #algo = ABCAlgo_Elite_2(pm, 100, 100, 100, 1, 5)
    #algo = ABCAlog_Adaptive(pm, 100, 100, 50, 1)
    #algo = ABCAlgo_Islands(pm, 10, 10, 10, 1, 20, 5)
    #algo = ABCAlgo_Islands_Mixer(pm, 10, 10, 10, 1, 10, 2)
    #algo.slove()
    #algo.show_result()
    #exit()

    algo_class_lst = [
        #ABCAlgo,
        ABCAlgo_Elite,
        #ABCAlgo_Elite_2,
        ABCAlog_Adaptive,
        ABCAlgo_Islands,
        ABCAlgo_Islands_Mixer,
    ]

    iter_number_lst = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] + [ (i+2) * 100 for i in range(4) ]
    sample_number = 50
    result_table = []

    for algo_class in algo_class_lst:
        print(algo_class.__name__)
        best_val_lst = []
        slove_time_lst = []
        for iter_max_num in iter_number_lst:
            #print('    -> tier number : {:d}'.format(iter_max_num))
            sum_val = 0
            sum_time = 0
            for i in range(sample_number):
                #print('        -> sample No.[{:d}]'.format(i))
                algo = get_algo_class(algo_class, pm, iter_max_num)

                time_start = time.time()
                algo.slove()
                slove_time = time.time() - time_start

                sum_val += algo.best_resolution.fitness
                sum_time += slove_time

            best_val_lst.append(sum_val / sample_number)
            print('    slove time : {:.2f}s'.format(sum_time / sample_number))
        #plt.plot(iter_number_lst, best_val_lst, marker='o', label=algo_class.__name__)
        result_table.append(best_val_lst)
        #plt.xticks(iter_number_lst)

    for iter_max_num in iter_number_lst:
        print('{:6d}\t'.format(iter_max_num), end='|')
    print('')
    for algo_class, best_val_lst in zip(algo_class_lst, result_table):
        for val in best_val_lst:
            print('{:6.2f}\t'.format(val), end='|')
        print(' -> {}'.format(algo_class.__name__))
    #plt.legend()
    #plt.show()

