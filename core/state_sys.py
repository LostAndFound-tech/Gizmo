

class StateSys():

    cur_state = None
    last_state = None
    def __init__(self):
        self.cur_state = None
        self.last_state = None

    def current_state(self):
        return {"current": self.cur_state, "Last":self.last_state}
    
    def change_state(self, newState):
        self.last_state = self.cur_state
        self.cur_state = newState

    