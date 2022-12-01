from turtle import width
import Block as bl
import Space as sp
import numpy as np
import operator
import datetime
import copy
import pandas as pd
import random
import scipy.stats as stats
from queue import Queue
from collections import deque
from base import BaseEnv

class _BlockArrangement(BaseEnv):

    def __init__(
        self,
        name,
        stock_scale,
        utilization,
        width, 
        height, 
        block_size,
        num_blocks,
        provided_ratio,
        entrance_cnt,
        entrance_position,
        rearrangement_occur_reward,
        arrangement_fail_reward,
        schedule_clear_reward_without_rearrangement,
        **kwargs,
    ):
        self.width = width
        self.height = height
        self.block_size = block_size
        self.NUM_BLOCKS = num_blocks
        self.arrival_scale = self.find_block_arrival_term(stock_scale, utilization)
        self.stock_scale = stock_scale
        self.state_size = self.width * self.height + 2
        self.action_size = (self.width-1) * self.height
        self.action_type = "discrete"
        self.provided_ratio = provided_ratio
        self.entrance_cnt = entrance_cnt
        self.entrance_position = entrance_position
        self.rearrangement_occur_reward = rearrangement_occur_reward
        self.arrangement_fail_reward = arrangement_fail_reward
        self.schedule_clear_reward_without_rearrangement = schedule_clear_reward_without_rearrangement
        self.LOGS = []
        self.name = str(name)
        self.idx = 0
        self._initialize()

    def _initialize(self):
        space, blocks = self.generate_space_block(self.width, self.height, self.NUM_BLOCKS, self.arrival_scale, self.stock_scale, self.provided_ratio)
        self.BLOCKS = sorted(blocks, key=operator.attrgetter('_startdate'))
        self.SPACE = space
        self.SPACE.update_blocks(copy.deepcopy(self.BLOCKS))
        self.STAGE = 0

    def reset(self):
        self._initialize()
        status = self.SPACE.get_status(0).flatten()
        status = np.append(status, np.array([0,0]))
        return np.expand_dims(status, 0)

    def step(self, action):
        reward = 0
        done = False
        blocks = self.SPACE.get_blocks()
        target = blocks[self.STAGE]
        x_loc = int(action / self.height)
        y_loc = int(action % self.height)
        is_arrangible = self.SPACE.arrangement(self.STAGE, x_loc, y_loc, target.isin)

        self.STAGE += 1
        if not is_arrangible:
            reward = self.arrangement_fail_reward
            done = True
        elif self.STAGE == len(blocks):
            reward = self.substep()
            if reward == 0:
                done = True
                if self.STAGE == self.NUM_BLOCKS:
                    reward = self.schedule_clear_reward_without_rearrangement
        else:
            reward = self.substep()


        status = self.get_state().flatten()
        if not done:
            new_blocks = self.SPACE.get_blocks()
            next_block = new_blocks[self.STAGE]
            if next_block.isin == True:
                status = np.append(status, np.array(next_block.get_location()))
            else:
                status = np.append(status, np.array(self.entrance_position))
        else:
            status = np.append(status, np.array([0,0]))

        next_state, reward, done = map(
            lambda x: np.expand_dims(x, 0), [status, [reward], [done]]
        )

        return (next_state, reward, done)

    def substep(self):
        current = self.SPACE.event_in[self.STAGE - 1]
        next = datetime.datetime(datetime.MAXYEAR, 1, 1)
        if len(self.SPACE.event_in) != self.STAGE:
            next = self.SPACE.event_in[self.STAGE]

        #현재 적치와 다음 적치 사이에 발생하는 Move_OUT 블록 추출
        cnt = 0
        transfers = []
        out_events = sorted(self.SPACE.event_out, key=lambda out: out[0])
        for i in range(self.STAGE):
            if current < out_events[i][0] <= next:
                transfers.append(out_events[i])
        if len(transfers) == 0:
            return 0
        
        current_blocks = []
        blocks = self.SPACE.get_blocks()
        
        for block in blocks:
            start, end = block.get_schedule()
            if start <= current < end:
                current_blocks.append(block)
        transfers = self.move_out_order_optimization(transfers)
        for transfer in transfers:
            state = self.SPACE.get_status(max(0, self.STAGE - 1))
            x_loc, y_loc = transfer[1].get_location()

            #Move_OUT 가능한지 체크
            possible = self.reachableToExit(state, x_loc, y_loc)
            if possible:
                state[x_loc][y_loc] = -1.0
                self.BLOCKS = blocks
            else:
                #간섭블록 위치 가져오기
                blocking_position = self.find_blocking_block(state, x_loc, y_loc)
                #간섭블록의 위치로 간섭블록 탐색
                for block in self.BLOCKS:
                    id = block.name
                    isin = block.isin
                    start, end = block.get_schedule()

                new_blocks = copy.deepcopy(blocks)
                for block in blocks:
                    if block.isin == True:
                        x, y = block.get_location()
                        for pos in blocking_position:
                            if x == int(pos / self.height) and y == int(pos % self.height):
                                cnt += self.rearrangement_occur_reward
                                block._startdate = blocks[self.STAGE-1]._startdate
                                new_blocks.insert(self.STAGE, block)
                self.BLOCKS = new_blocks
                break
        
        self.SPACE.update_blocks(copy.deepcopy(self.BLOCKS))
        return cnt


    def close(self):
        pass

    def find_block_arrival_term(self, stock_scale, utilization):
        arrival_scale = stock_scale*24 / utilization * 100 / ((self.width-1) * self.height) 
        return arrival_scale

    def move_out_order_optimization(self, transfers):
        new_transfers = []
        q = deque()
        q.append([self.width-1, 0])
        dx = [-1,+1,0,0]
        dy = [0,0,-1,+1]
        visit = np.full((self.width, self.height), 0, dtype = int)
        for i in range(0, self.height):
            visit[self.width-1][i] = 1
        while q:
            cur_x, cur_y = q.popleft()

            for i in range(4):
                next_x = cur_x + dx[i]
                next_y = cur_y + dy[i]
                if next_x < 0 or next_x >= self.width or next_y < 0 or next_y >= self.height:
                    continue
                if visit[next_x][next_y] == 0:
                    visit[next_x][next_y] = 1
                    q.append([next_x, next_y])
                    for transfer in transfers:
                        x, y = transfer[1].get_location()
                        if x == next_x and y == next_y:
                            new_transfers.append(transfer)

        return new_transfers

    def find_shortest_path(self, state, stack, visit, shortest_len, breaking):
        cur, path = stack.pop()

        if cur == ((self.width-1)*self.height):
            return path

        cx = int(cur / self.height)
        cy = int(cur % self.height)
        dx = [1,-1,0,0]
        dy = [0,0,1,-1]
        for i in range(4):
            nx = cx + dx[i]
            ny = cy + dy[i]
            if nx < 0 or nx >=self.width or ny < 0 or ny >= self.height:
                continue
            if nx == self.height and ny != 0:
                continue 
            if breaking <= shortest_len and (visit[nx][ny] > breaking or visit[nx][ny] == -1):
                path.append(nx * self.height + ny)
                visit[nx][ny] = breaking
                stack.append((nx * self.height + ny, path))
                if state[nx][ny] != -1:
                    finish = self.find_shortest_path( state, stack, visit, shortest_len, breaking + 1)
                else:
                    finish = self.find_shortest_path( state, stack, visit, shortest_len, breaking)
                if finish != -1:
                    return finish
                path.pop()
                
        return -1

    def find_blocking_block(self, state, xloc, yloc):
        graph = [[] for i in range(self.width * self.height)]
        dx = [1, -1, 0, 0]
        dy = [0, 0, 1, -1]
        for x in range(self.width):
            for y in range(self.height):
                node = x * self.height + y
                for d in range(4):
                    nx = x + dx[d]
                    ny = y + dy[d]
                    if nx<0 or nx>=self.width or ny<0 or ny>=self.height:
                        continue
                    if nx == self.width and ny != 0:
                        continue
                    adj_node = nx * self.height + ny
                    if state[x][y] == -1:
                        graph[adj_node].append((node, 0))
                    else:
                        graph[adj_node].append((node, 1))

        target_node = xloc * self.height + yloc
        distance = [int(1e9)] * (self.width * self.height)
        distance[target_node] = 0
        visited = [False] * (self.width * self.height)
        visited[target_node] = True

        for i in graph[target_node]: distance[i[0]] = i[1]
        for _ in range(self.width * self.height - (self.height - 1)):
            idx = 0
            min_dist = int(1e9)
            for j in range(self.width * self.height - (self.height - 1)):
                if distance[j] < min_dist and not visited[j]:
                    min_dist = distance[j]
                    idx = j

            visited[idx] = True
            for adj_node in graph[idx]:
                cost = min_dist + adj_node[1]
                if cost < distance[adj_node[0]]:
                    distance[adj_node[0]] = cost

        shortest_len = distance[(self.width-1) * self.height]

        visit = np.full((self.width,self.height), -1, dtype = int)
        st = []
        st.append((target_node, []))
        visit[xloc][yloc] = 0
        for i in range(1, self.height): 
            visit[self.width-1][i] = 10
        shortest_path = self.find_shortest_path(state, st, visit, shortest_len, 0)
        
        blocking_blocks = []
        for i in shortest_path:
            x = int(i/self.height)
            y = int(i%self.height)
            if state[x][y] != -1:
                blocking_blocks.append(i)

        return blocking_blocks

    def reachableToExit(self, state, x, y):
        q = deque()
        q.append([x, y])
        dx = [-1,+1,0,0]
        dy = [0,0,-1,+1]
        visit = np.full((self.width, self.height), 0, dtype = int)
        visit[x,y] = 1
        for i in range(1, self.height):
            visit[self.width-1][i] = 1
        is_reachable = False
        while q:
            cur_x, cur_y = q.popleft()
            if cur_x == self.width-1 and cur_y == 0:
                is_reachable = True
                break
            for i in range(4):
                next_x = cur_x + dx[i]
                next_y = cur_y + dy[i]
                if next_x < 0 or next_x >= self.width or next_y < 0 or next_y >= self.height:
                    continue
                if state[next_x][next_y] == -1  and visit[next_x][next_y] == 0:
                    visit[next_x][next_y] = 1
                    q.append([next_x, next_y])
        return is_reachable
    
    def generate_space_block(self, width, height, num_block, arrival_scale, stock_scale, provided_ratio):
        space = sp.Space(width, height)
        new_schedule = self.generate_schedule(num_block, arrival_scale, stock_scale, provided_ratio)
        self.idx += 1
        blocks = []
        for index, row in new_schedule.iterrows():
            date_in = row['Move_IN']
            date_out = row['Move_OUT']
            term = row['duration']
            provided = row['provided']
            blocks.append(bl.Block(1,1, date_in, date_out, term, row['BlockID'], provided))
        return space, blocks

    def generate_schedule(self, num_block, arrival_scale, stock_scale, provided_ratio):
        new_schedule = pd.DataFrame(columns=['BlockID','Move_IN','Move_OUT','duration','JIBUN', 'provided'])
        arrivals = stats.poisson.rvs(arrival_scale, size=num_block) 
        stocks = stats.poisson.rvs(stock_scale, size=num_block)

        provided_cnt = int(num_block * provided_ratio)
        provided_blocks = sorted(random.sample(range(num_block), provided_cnt))
        provided_idx = 0

        current_time = datetime.datetime.strptime('2022-05-01 09:00:00', '%Y-%m-%d %H:%M:%S')
        idx = 0
        for i in range(num_block):
            next_time = current_time + datetime.timedelta(hours=int(arrivals[i]))
            end_time = next_time + datetime.timedelta(days=int(stocks[i]))
            duration = datetime.timedelta(days=end_time.day - next_time.day) + datetime.timedelta(days=(end_time.month - next_time.month)*30)
            current_time = next_time
            if duration.days == 0:
                duration = datetime.timedelta(days=1)
                end_time += duration
            
            next_time = datetime.datetime(next_time.year, next_time.month, next_time.day, next_time.hour)
            end_time = datetime.datetime(end_time.year, end_time.month, end_time.day)

            provided = False
            if provided_idx < provided_cnt and idx == provided_blocks[provided_idx]:
                provided = True
                provided_idx += 1

            row = pd.Series([idx, next_time, end_time, duration.days, '', provided],
                    index=['BlockID', 'Move_IN', 'Move_OUT', 'duration', 'JIBUN', 'provided'])
            idx += 1
            new_schedule = new_schedule.append(row, ignore_index=True)
        
        return new_schedule

    def get_state(self):
        state = self.SPACE.get_status(max(0, self.STAGE - 1))
        return state

class block_arrangement(_BlockArrangement):
    def __init__(self, **kwargs):
        super(block_arrangement, self).__init__(f"block_arrangement", **kwargs)