#import objects.Block
import numpy as np
from copy import deepcopy
import datetime
from collections import deque

class Space:

    def __init__(self, width, height, name):
        self.width=width
        self.height=height
        self.name=name
        self.RESULTS=[]
        self.status = np.full((self.width, self.height), -1.0, dtype=float)
    
    def update_blocks(self, blocks):
        events = []
        event_in=[]
        event_out=[]
        for block in blocks:
            start, end = block.get_schedule()
            events.append(start)
            events.append(end)
            event_in.append(start)
            event_out.append([end, block])
        if isinstance(events[0], datetime.date):
            self.TIMETYPE = datetime.date
        else :
            self.TIMETYPE = int
        events = list(set(events))
        events.sort()
        self.EVENTS = events
        self._blocks=blocks
        self.event_in=event_in
        self.event_out=event_out

    def update_status(self, stage):
        if stage!=0:
            self._transfer_blocks(stage)
        block=self._blocks[stage]
        r=0
        block.isin = True
        width, height = block.get_dimension()
        xloc, yloc = block.get_location()
        bounds = self._make_boundary([xloc, yloc], [width, height])
        for bound in bounds:
            if self.status[bound[0], bound[1]] == 1:
                r += 1
        for i in range(width):
            for j in range(height):
                if (xloc + i < self.status.shape[0] and yloc + j < self.status.shape[1]):
                    self.status[xloc + i, yloc + j] += 1.0
                else:
                    if (i == 0):
                        yloc += -1
                        self.status[xloc + i, yloc + j] += 1.0
                    else:
                        xloc += -1
                        self.status[xloc + i, yloc + j] += 1.0
        self.RESULTS.append(deepcopy(self.status))
        arrangible = True
        for a in np.nditer(self.status):
            if (a > 1):
                arrangible = False
                r=0
                break
        return arrangible, r

    def reachable(self, state, cur_x, cur_y, next_x, next_y):
        q = deque()
        q.append([cur_x, cur_y])
        dx = [-1,+1,0,0]
        dy = [0,0,-1,+1]
        visit = np.full((6,5), 0, dtype = int)
        visit[cur_x,cur_y] = 1
        is_reachable = False
        while q:
            cx, cy = q.popleft()
            if cx == next_x and cy == next_y:
                is_reachable = True
                break
            for i in range(4):
                nx = cx + dx[i]
                ny = cy + dy[i]
                if nx < 0 or nx >= 6 or ny < 0 or ny >= 5:
                    continue
                if state[nx][ny] != -2 and visit[nx][ny] == 0:
                    visit[nx][ny] = 1
                    q.append([nx, ny])

        return is_reachable

    def arrangement(self, stage, next_x, next_y, rearrange):
        state = np.full((self.width, self.height), -1.0, dtype=float)
        current_day = self.event_in[stage]
        block = self._blocks[stage]

        if rearrange == True:
            cur_x, cur_y = block.get_location()
        else:
            cur_x, cur_y = 5, 0

        # print(f"cur_x, cur_y: {cur_x}, {cur_y}")
        # print(f"next_x, next_y: {next_x}, {next_y}")

        if stage!=0:
            blocks = self.get_blocks()
            for i in range(stage):
                _block = blocks[i]
                _start, _end = _block.get_schedule()
                if _start <= current_day < _end:
                    _x, _y = _block.get_location()
                    state[_x, _y] = (_end - current_day).days
        
        # print(f"reachable: {self.reachable(state, cur_x, cur_y, next_x, next_y)}")
        # print(f"state[next_x, next_y] == {state[next_x, next_y]}")
        #empty position check
        if state[next_x, next_y] == -1.0 and self.reachable(state, cur_x, cur_y, next_x, next_y):
            arrangible = True
            state[cur_x][cur_y] = -1
            state[next_x, next_y] = block.term
            block.set_location(next_x, next_y)
            block.isin = True
            self._blocks[stage] = block
        else:
            arrangible = False
            # print(f"rearrange = {rearrange}")
            # print(f"action(x,y) = {cur_x} {cur_y}")
            # for i in range(6):
            #     print(state[i])
            # print()
            state[next_x, next_y] = -2.0
            # for i in range(6):
            #     print(state[i])
            # print()
            # print()
        
        self.RESULTS.append(state)
       
        return arrangible

    def separate_area(self, state, location=[0,0], exit=0):
        if exit==1:
            location = [location[1], state.shape[0]-location[0]-1]
        elif exit==2:
            location = [state.shape[0]-location[0]-1, state.shape[1]-location[1]-1]
        elif exit==3:
            location = [state.shape[1] - location[1]-1, location[0]]
        state = np.rot90(state, -exit)
        exitside=[]
        otherside=[]
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                if state[i,j] != -1 and j != location[1]:
                    if j<location[1]:
                        exitside.append(state[i,j])
                    else:
                        otherside.append(state[i,j])
        return exitside, otherside

    def _transfer_blocks(self, stage):
        for day, block in self.event_out:
            if day>self.event_in[stage-1] and day<=self.event_in[stage]:
                width, height = block.get_dimension()
                xloc, yloc = block.get_location()
                for i in range(width):
                    for j in range(height):
                        if (xloc + i < self.status.shape[0] and yloc + j < self.status.shape[1]):
                            self.status[xloc + i, yloc + j] = 0.0
                        else:
                            if (i == 0):
                                yloc += -1
                                self.status[xloc + i, yloc + j] = 0.0
                            else:
                                xloc += -1
                                self.status[xloc + i, yloc + j] = 0.0

    def set_status(self, date):
        self.CURRENTDATE = date
        status = np.full((self.status.shape[0], self.status.shape[1]), .0, dtype=float)
        for block in self._blocks:
            startdate, enddate = block.get_schedule()
            if (date >= startdate and date < enddate):
               width, height = block.get_dimension()
               xloc, yloc = block.get_location()
               for i in range(width):
                   for j in range(height):
                       if (yloc + j < status.shape[0] and xloc + i < status.shape[1]):
                           status[yloc + j, xloc + i] += 1.0
                       else:
                           if (j == 0):
                               xloc += -1
                               status[yloc + j, xloc + i] += 1.0
                           else:
                               yloc += -1
                               status[yloc + j, xloc + i] += 1.0
        self.RESULTS.append(status)
        arrangible = True
        for a in np.nditer(status):
            if (a > 1):
                arrangible = False
                break
        return arrangible

    def get_status(self, stage):
        if (len(self.RESULTS) == 0):
            status = np.full((self.width, self.height), -1.0, dtype=float)
        else:
            status = self.RESULTS[stage]
        return status

    def get_blocks(self):
        return self._blocks

    def _make_boundary(self, location, size):
        bounds=[]
        if location[1] != 0:
            for i in range(size[0]):
                if location[0] + i< self.width:
                    bounds.append([location[0]+i, location[1]-1])
        if location[0] + size[0] < self.width:
            for i in range(size[1]):
                if location[1] + i <self.height:
                    bounds.append([location[0] + size[0], location[1]+i])
        if location[1] + size[1] < self.height:
            for i in range(size[0]):
                if location[0] + i < self.width:
                    bounds.append([location[0]+i, location[1] + size[1]])
        if location[0] != 0:
            for i in range(size[1]):
                if location[1] + i < self.height:
                    bounds.append([location[0]-1, location[1]+i])
        return bounds

    def modify_latest_state(self, new_state):
        if len(self.RESULTS) > 0:
            self.RESULTS[-1] = new_state











































