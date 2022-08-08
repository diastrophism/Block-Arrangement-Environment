from turtle import width
import Block as bl
import Space as sp
import numpy as np
import operator
import datetime
import copy
import pandas as pd
import scipy.stats as stats
from queue import Queue
from collections import deque
from base import BaseEnv

class _BlockArrangement(BaseEnv):

    def __init__(
        self,
        name,
        arrival_scale,
        stock_scale,
        width=6, 
        height=5, 
        num_blocks=50,
        **kwargs,
    ):
        space, blocks = self.generate_space_block(width, height, num_blocks, arrival_scale, stock_scale)
        self.NUM_BLOCKS = num_blocks
        self.BLOCKS = sorted(blocks, key=operator.attrgetter('_startdate'))
        self.height = space.height
        self.width = space.width
        self.arrival_scale = arrival_scale
        self.stock_scale = stock_scale
        self.space_name = space.name
        self.state_size = self.width * self.height
        self.action_size = (self.width-1) * self.height
        self.action_type = "discrete"
        self.LOGS = []
        self.cumulate = np.zeros([self.width, self.height])
        self.name = str(name)
        self._initialize()

        self.x_loc = [0,0,0,0,0,
                      1,1,1,1,1,
                      2,2,2,2,2,
                      3,3,3,3,3,
                      4,4,4,4,4,
                      5]

        self.y_loc = [0,1,2,3,4,
                      0,1,2,3,4,
                      0,1,2,3,4,
                      0,1,2,3,4,
                      0,1,2,3,4,
                      0]

    def _initialize(self):
        space, blocks = self.generate_space_block(self.width, self.height, self.NUM_BLOCKS, self.arrival_scale, self.stock_scale)
        self.BLOCKS = sorted(blocks, key=operator.attrgetter('_startdate'))
        self.space_name = space.name
        self.SPACE = sp.Space(self.width, self.height, self.space_name)
        self.SPACE.update_blocks(copy.deepcopy(self.BLOCKS))
        self.STAGE = 0
        #print(f"episode Start!")
        #print(f"state size: {self.state_size}")
        #print(f"action size: {self.action_size}")

    def reset(self):
        self._initialize()
        status = self.SPACE.get_status(0)
        status = status.flatten()
        status = np.expand_dims(status, 0)
        # print("--------------reset--------------")
        # print()
        # print()
        return status

    def step(self, action):
        reward = 0
        done = False
        blocks = self.SPACE.get_blocks()
        target = blocks[self.STAGE]

        x_loc = int(action / self.height)
        y_loc = int(action % self.height)
        
        is_arrangible = self.SPACE.arrangement(self.STAGE, x_loc, y_loc, target.isin)
        # print(f"is_arrangible: {is_arrangible}")

        self.STAGE += 1
        # print(f"STAGE: {self.STAGE}")
        # print(f"action : {action}")
        s = self.get_state()
        # print("적치 후")
        # for i in range(6):
        #     print(s[i])

        if not is_arrangible:
            reward = -10
            done = True
        elif self.STAGE == len(blocks):
            reward = +10
            done = True
            self.substep()
            #print("----------episode finish------------")
        else:
            #reward = +1
            self.substep()

        status = self.get_state().flatten()
        next_state, reward, done = map(
            lambda x: np.expand_dims(x, 0), [status, [reward], [done]]
        )
        
        # idx = 0
        # for i in self.SPACE.get_blocks():
        #     if i.isin == True:
        #         print(f"idx: {idx}, {i.name}, {i.get_schedule()}, {i.term}, {i.get_location()}")
        #     else:
        #         print(f"idx: {idx}, {i.name}, {i.get_schedule()}, {i.term}")
        #     idx = idx + 1

        return (next_state, reward, done)

    def substep(self):
        current = self.SPACE.event_in[self.STAGE - 1]
        next = datetime.datetime(datetime.MAXYEAR, 1, 1)
        
        if len(self.SPACE.event_in) != self.STAGE:
            next = self.SPACE.event_in[self.STAGE] #다음 적치 날짜

        #현재 적치와 다음 적치 사이에 발생하는 Move_OUT 블록 추출
        transfers = []
        out_events = sorted(self.SPACE.event_out, key=lambda out: out[0])
        for i in range(self.STAGE):
            if current < out_events[i][0] <= next:
                transfers.append(out_events[i])
        if len(transfers) == 0:
            return +1
        
        #print("반출 작업 발생!!")
        
        current_blocks = []
        blocks = self.SPACE.get_blocks()
        for block in blocks:
            start, end = block.get_schedule()
            if start <= current < end:
                current_blocks.append(block)
        
        cnt = 0
        for transfer in transfers:
            state = self.SPACE.get_status(max(0, self.STAGE - 1))
            x_loc, y_loc = transfer[1].get_location()

            #Move_OUT 가능한지 체크
            possible = self.reachableToExit(state, x_loc, y_loc)
            #print(f"Move_OUT possible: {possible}")
            if possible:
                state[x_loc][y_loc] = -1.0
                self.BLOCKS = blocks
            else:
                #print("비작업 블록 발생!!")
                #간섭블록의 위치를 가져옴
                blocking_position = self.find_blocking_block(state, x_loc, y_loc)
                #print(f"blocking_postion: {blocking_position}")
                #간섭블록의 위치로 간섭블록 탐색
                new_blocks = copy.deepcopy(blocks)
                for block in blocks:
                    #print(block.name)
                    if block.isin == True:
                        x, y = block.get_location()
                        for pos in blocking_position:
                            if x == int(pos / self.height) and y == int(pos % self.height):
                                cnt = cnt + 1
                                block._startdate = blocks[self.STAGE-1]._startdate
                                new_blocks.insert(self.STAGE, block)
                                #print(f"blockID: {block.name}")
                self.BLOCKS = new_blocks
                #self.NUM_BLOCKS = len(new_blocks)
                break

        # s = self.get_state()
        # print("반출 후")
        # for i in range(6):
        #     print(s[i])
        
        self.SPACE.update_blocks(copy.deepcopy(self.BLOCKS))
        return -cnt


    def close(self):
        pass

    def find_shortest_path(self, state, stack, visit, shortest_len, breaking):
        cur, path = stack.pop()
        #print(f"cur: {cur}")
        #print(f"path: {path}")
        if cur == 25:
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
            if (nx == 5 and ny == 1) or (nx == 5 and ny == 2) or (nx == 5 and ny == 3) or (nx == 5 and ny == 4):
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
                #visit[i[0]] = False
                
        return -1

    def find_blocking_block(self, state, xloc, yloc):
        INF = int(1e9)
        distance = [INF] * 27
        visited = [False] * 27
        graph = [[] for i in range(27)]
        
        for i in range(4):
            for j in range(4):
                a = int(i * self.height) + j
                b = int(i * self.height) + j + 1
                if state[i][j] != -1:
                    graph[b].append((a, 1))
                else:
                    graph[b].append((a, 0))
                
                if state[i][j+1] != -1:
                    graph[a].append((b, 1))
                else:
                    graph[a].append((b, 0))
                
                c = (i+1)*5 + j
                if state[i][j] != -1:
                    graph[c].append((a, 1))
                else:
                    graph[c].append((a, 0))
                
                if state[i+1][j] != -1:
                    graph[a].append((c, 1))
                else:
                    graph[a].append((c, 0))
        
        for i in range(4):
            a = 20 + i
            b = 20 + i + 1
            if state[4][i+1] != -1:
                graph[a].append((b, 1))
            else:
                graph[a].append((b, 0))
            
            if state[4][i] != -1:
                graph[b].append((a, 1))
            else:
                graph[b].append((a, 0))
        
            a = 4 + i*5
            b = 4 + (i+1)*5
            if state[i+1][4] != -1:
                graph[a].append((b, 1))
            else:
                graph[a].append((b, 0))
            
            if state[i][4] != -1:
                graph[b].append((a, 1))
            else:
                graph[b].append((a, 0))
                
        graph[20].append((25,0))
        graph[25].append((20,0))
        
        start = xloc * 5 + yloc
        distance[start] = 0
        visited[start] = True
        
        for i in graph[start]:
            distance[i[0]] = i[1]
            
        for i in range(26):
            smallest = INF
            index = 0
            for j in range(26):
                if distance[j] < smallest and not visited[j]:
                    smallest = distance[j]
                    index = j
            
            visited[index] = True
            for j in graph[index]:
                cost = distance[index] + j[1]
                if cost < distance[j[0]]:
                    distance[j[0]] = cost
        
        shortest_len = distance[25]
        #print(shortest_len)
        visit = np.full((6,5), -1, dtype = int)
        st = []
        st.append((start, []))
        x = int(start/self.height)
        y = int(start%self.height)
        visit[x][y] = 0
        visit[5][1] = 10
        visit[5][2] = 10
        visit[5][3] = 10
        visit[5][4] = 10
        shortest_path = self.find_shortest_path(state, st, visit, shortest_len, 0)
        
        blocking_blocks = []
        #print(shortest_path)
        
        for i in shortest_path:
            x = int(i/self.height)
            y = int(i%self.height)
            if state[x][y] != -1:
                blocking_blocks.append(i)

        # for i in range(len(shortest_path) - 1):
        #     if distance[shortest_path[i]] != distance[shortest_path[i+1]]:
        #         blocking_blocks.append(shortest_path[i+1])

        return blocking_blocks

    def reachableToExit(self, state, xloc, yloc):
        q = deque()
        q.append([xloc, yloc])

        dx = [-1,+1,0,0]
        dy = [0,0,-1,+1]
        visit = np.full((6,5), 0, dtype = int)
        visit[xloc,yloc] = 1
        visit[5][1] = 10
        visit[5][2] = 10
        visit[5][3] = 10
        visit[5][4] = 10
        is_reachable = False

        while q:
            cur_x, cur_y = q.popleft()
                        
            if cur_x == 5 and cur_y == 0:
                is_reachable = True
                break

            for i in range(4):
                next_x = cur_x + dx[i]
                next_y = cur_y + dy[i]

                if next_x < 0 or next_x >= 6 or next_y < 0 or next_y >= 5:
                    continue

                if state[next_x][next_y] == -1 and visit[next_x][next_y] == 0:
                    visit[next_x][next_y] = 1
                    q.append([next_x, next_y])

        return is_reachable

    
        
    def generate_space_block(self, width, height, num_block, arrival_scale, stock_scale):
        space = sp.Space(width, height, 'test_area')
        new_schedule = self.generate_schedule(num_block, arrival_scale, stock_scale)
        blocks = []
        for index, row in new_schedule.iterrows():
            date_in = row['Move_IN']
            date_out = row['Move_OUT']
            term = row['duration']
            blocks.append(bl.Block(1,1, date_in, date_out, term, row['BlockID']))
        return space, blocks

    def generate_schedule(self, num_block, arrival_scale, stock_scale):
        new_schedule = pd.DataFrame(columns=['BlockID','Move_IN','Move_OUT','duration','JIBUN'])

        # rvs = random vairable samplit
        # scale = standard deviation
        arrivals = stats.expon.rvs(scale=arrival_scale, size=num_block) 
        stocks = stats.expon.rvs(scale=stock_scale, size=num_block)
        current_time = datetime.datetime.strptime('2022-05-01 09:00:00', '%Y-%m-%d %H:%M:%S')
        j = 0
        for i in range(num_block):
            next_time = current_time + datetime.timedelta(hours=arrivals[i]) # 평균적으로 대략 4.5 시간 
            end_time = next_time + datetime.timedelta(days=stocks[i])
            duration = datetime.timedelta(days=end_time.day - next_time.day) + datetime.timedelta(days=(end_time.month - next_time.month)*30)
            current_time = next_time
            if duration.days == 0:
                duration = datetime.timedelta(days=1)
                end_time += duration
            
            next_time = datetime.datetime(next_time.year, next_time.month, next_time.day, next_time.hour)
            end_time = datetime.datetime(end_time.year, end_time.month, end_time.day)
            j += 1
            row = pd.Series(['block' + str(j), next_time, end_time, duration.days, ''],
                    index=['BlockID', 'Move_IN', 'Move_OUT', 'duration', 'JIBUN'])
            new_schedule = new_schedule.append(row, ignore_index=True)

        return new_schedule

    def get_state(self):
        state = self.SPACE.get_status(max(0, self.STAGE - 1))
        #state = self.normalize_state(state, self.STAGE)
        return state

    def normalize_state(self, state, stage):
        norm_state = np.array(state)
        blocks = self.SPACE.get_blocks()
        if len(blocks) == stage:
           stage += -1
        duration = blocks[stage].term
        for i in range(norm_state.shape[0]):
            for j in range(norm_state.shape[1]):
                if norm_state[i, j] != -1.0:
                    norm_state[i, j] = norm_state[i, j] / duration
                if norm_state[i, j] >= 3:
                    norm_state[i, j] = 3.0
        return norm_state

    def set_transporter(self, transporter):
        self.Transporter = transporter

    def get_reward(self, state, target):
        blocks = []
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                if state[i, j] != -1.0:
                    blocks.append(np.array([j, i]))
        moves, _ = self.Transporter.get_block_moves(blocks, target)
        reward = max(0, 3 - moves)
        return reward

    def transport_block(self, state, target, current_blocks):
        blocks = []
        terms = []
        index_curr = []
        index = -1
        index_target = -1
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                if state[i, j] != -1.0:
                    index += 1
                    blocks.append(np.array([i, j]))
                    terms.append(state[i, j])
                    if i == target[0] and j == target[1]:
                        index_target = index
                    for k in range(len(current_blocks)):
                        x, y = current_blocks[k].get_location()
                        if x == i and y == j:
                            index_curr.append(k)
        if index_target == -1:
            print('here')
        moves, moved_blocks = self.Transporter.get_block_moves(blocks, index_target, self.name)
        moved_state = np.full(state.shape, -1.0, dtype=float)
        try:
            for i in range(len(moved_blocks)):
                if i == index_target:
                    continue
                current_blocks[index_curr[i]].set_location(moved_blocks[i][0], moved_blocks[i][1])
                moved_state[moved_blocks[i][0], moved_blocks[i][1]] = terms[i] - terms[index_target]

            self.SPACE.modify_latest_state(moved_state)
            self.LOGS.append(self.SPACE.RESULTS[-1])
            del current_blocks[index_curr[index_target]]
        except:
            print('here')
        if moves == 0:
            reward = 2
        else:
            reward = max(0, 1/moves)
        print(moves)
        return reward

class block_arrangement(_BlockArrangement):
    def __init__(self, **kwargs):
        super(block_arrangement, self).__init__(f"block_arrangement", **kwargs)



























