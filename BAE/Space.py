#import objects.Block
import numpy as np
from copy import deepcopy
import datetime
from collections import deque

class Space:

    def __init__(self, width, height):
        self.width=width
        self.height=height
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

    def reachable(self, state, cur_x, cur_y, next_x, next_y):
        q = deque()
        q.append([cur_x, cur_y])
        dx = [-1,+1,0,0]
        dy = [0,0,-1,+1]
        visit = np.full((self.width, self.height), 0, dtype = int)
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
                if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                    continue
                if nx == self.width-1 and ny != 0:
                    continue
                if state[nx][ny] == -1 and visit[nx][ny] == 0:
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
            cur_x, cur_y = self.width-1, 0

        if stage!=0:
            blocks = self.get_blocks()
            for i in range(stage):
                _block = blocks[i]
                _start, _end = _block.get_schedule()
                if _start <= current_day < _end:
                    _x, _y = _block.get_location()
                    if _block.provided:
                        state[_x, _y] = (_end - current_day).days
                    else:
                        state[_x, _y] = -3

        if (state[next_x, next_y] == -1) and self.reachable(state, cur_x, cur_y, next_x, next_y):
            arrangible = True
            state[cur_x][cur_y] = -1
            if block.provided:
                state[next_x, next_y] = block.term
            else:
                state[next_x, next_y] = -3
            
            block.set_location(next_x, next_y)
            block.isin = True
            self._blocks[stage] = block

            if rearrange == True:
                for i in range(stage):
                    pre = self._blocks[i]
                    if pre.name == block.name:
                        pre._enddate = block._startdate
                        self._blocks[i] = pre
                        break
        else:
            arrangible = False
            state[next_x, next_y] = -2.0
        
        self.RESULTS.append(state)
        return arrangible

    def get_status(self, stage):
        if (len(self.RESULTS) == 0):
            status = np.full((self.width, self.height), -1.0, dtype=float)
        else:
            status = self.RESULTS[stage]
        return status

    def get_blocks(self):
        return self._blocks

    def modify_latest_state(self, new_state):
        if len(self.RESULTS) > 0:
            self.RESULTS[-1] = new_state











































