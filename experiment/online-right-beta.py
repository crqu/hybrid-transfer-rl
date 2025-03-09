import numpy as np
import multiprocessing
from rlberry.agents import AgentWithSimplePolicy
import math
import numpy as np
import rlberry.spaces as spaces
from rlberry_research.envs.finite import GridWorld
from rlberry_research.rendering import Scene, GeometricPrimitive


# define the NRoom Environment
class NRoom(GridWorld):
    """
    GridWorld with N rooms of size L x L. The agent starts in the middle room.

    There is one small and easy reward in the first room,
    one big reward in the last room and zero reward elsewhere.

    There is a 5% error probability in the transitions when taking an action.

    Parameters
    ----------
    nrooms : int
        Number of rooms.
    reward_free : bool, default=False
        If true, no rewards are given to the agent.
    array_observation:
        If true, the observations are converted to an array (x, y)
        instead of a discrete index.
        The underlying discrete space is saved in env.discrete_observation_space.
    room_size : int
        Dimension (L) of each room (L x L).
    success_probability : double, default: 0.95
        Sucess probability of an action. A failure is going to the wrong direction.
    remove_walls : bool, default: False
        If True, remove walls. Useful for debug.
    initial_state_distribution: {'center', 'uniform'}
        If 'center', always start at the center.
        If 'uniform', start anywhere with uniform probability.
    include_traps: bool, default: False
        If true, each room will have a terminal state (a "trap").
    Notes
    -----
    The function env.sample() does not handle conversions to array states
    when array_observation is True. Only the functions env.reset() and
    env.step() are covered.
    """

    name = "N-Room"

    def __init__(
        self,
        nrooms=7,
        reward_free=False,
        array_observation=False,
        room_size=5,
        success_probability=0.95,
        remove_walls=False,
        initial_state_distribution="center",
        include_traps=False,
    ):
        assert nrooms > 0, "nrooms must be > 0"
        assert initial_state_distribution in ("center", "uniform")

        self.reward_free = reward_free
        self.array_observation = array_observation
        self.nrooms = nrooms
        self.room_size = room_size
        self.success_probability = success_probability
        self.remove_walls = remove_walls
        self.initial_state_distribution = initial_state_distribution
        self.include_traps = include_traps

        # Max number of rooms/columns per row
        self.max_rooms_per_row = 5

        # Room size (default = 5x5)
        self.room_size = room_size

        # Grid size
        self.room_nrows = math.ceil(nrooms / self.max_rooms_per_row)
        if self.room_nrows > 1:
            self.room_ncols = self.max_rooms_per_row
        else:
            self.room_ncols = nrooms
        nrows = self.room_size * self.room_nrows + (self.room_nrows - 1)
        ncols = self.room_size * self.room_ncols + (self.room_ncols - 1)

        # # walls
        walls = []
        for room_col in range(self.room_ncols - 1):
            col = (room_col + 1) * (self.room_size + 1) - 1
            for jj in range(nrows):
                if (jj % (self.room_size + 1)) != (self.room_size // 2):
                    walls.append((jj, col))

        for room_row in range(self.room_nrows - 1):
            row = (room_row + 1) * (self.room_size + 1) - 1
            for jj in range(ncols):
                walls.append((row, jj))

        # process each room
        start_coord = None
        terminal_state = None
        self.traps = []
        count = 0
        for room_r in range(self.room_nrows): 
            if room_r % 2 == 0:
                cols_iterator = range(self.room_ncols)
            else:
                cols_iterator = reversed(range(self.room_ncols))
            for room_c in cols_iterator:
                # existing rooms
                if count < self.nrooms:
                    # remove top wall
                    if ((room_c == self.room_ncols - 1) and (room_r % 2 == 0)) or (
                        (room_c == 0) and (room_r % 2 == 1)
                    ):
                        if room_r != self.room_nrows - 1:
                            wall_to_remove = self._convert_room_coord_to_global(
                                room_r, room_c, self.room_size, self.room_size // 2
                            )
                            if wall_to_remove in walls:
                                walls.remove(wall_to_remove)
                # rooms to remove
                else:
                    for ii in range(-1, self.room_size + 1):
                        for jj in range(-1, self.room_size + 1):
                            wall_to_include = self._convert_room_coord_to_global(
                                room_r, room_c, ii, jj
                            )
                            if (
                                wall_to_include[0] >= 0
                                and wall_to_include[0] < nrows
                                and wall_to_include[1] >= 0
                                and wall_to_include[1] < ncols
                                and (wall_to_include not in walls)
                            ):
                                walls.append(wall_to_include)
                    pass

                # start coord
                if count == nrooms // 2:
                    start_coord = self._convert_room_coord_to_global(
                        room_r, room_c, self.room_size // 2-1, self.room_size // 2-1
                    )
                # terminal state
                if count == nrooms - 1:
                    terminal_state = self._convert_room_coord_to_global(
                        room_r, room_c, self.room_size // 2+1, self.room_size // 2+1
                    )
                # trap
                if include_traps:
                    self.traps.append(
                        self._convert_room_coord_to_global(
                            room_r,
                            room_c,
                            self.room_size // 2 ,
                            self.room_size // 2 -1,
                        )
                    )
                    self.traps.append(
                        self._convert_room_coord_to_global(
                            room_r,
                            room_c,
                            self.room_size // 2 -1 ,
                            self.room_size // 2 ,
                        )
                    )
                    self.traps.append(
                        self._convert_room_coord_to_global(
                            room_r,
                            room_c,
                            self.room_size // 2,
                            self.room_size -1 ,
                        )
                    )
                count += 1

        terminal_states = (terminal_state,) + tuple(self.traps)

        if self.reward_free:
            reward_at = {}
        else:
            reward_at = {
                terminal_state: 1.0,
                start_coord: 0.01,
                (self.room_size // 2, self.room_size // 2): 0.1,
                (1,self.room_size -1):0.15
            }

        # Check remove_walls
        if remove_walls:
            walls = ()

        # Init base class
        GridWorld.__init__(
            self,
            nrows=nrows,
            ncols=ncols,
            start_coord=start_coord,
            terminal_states=terminal_states,
            success_probability=success_probability,
            reward_at=reward_at,
            walls=walls,
            default_reward=0.0,
        )

        # Check initial distribution
        if initial_state_distribution == "uniform":
            distr = np.ones(self.observation_space.n) / self.observation_space.n
            self.set_initial_state_distribution(distr)

        # spaces
        if self.array_observation:
            self.discrete_observation_space = self.observation_space
            self.observation_space = spaces.Box(0.0, 1.0, shape=(2,))

    def _convert_room_coord_to_global(
        self, room_row, room_col, room_coord_row, room_coord_col
    ):
        col_offset = (self.room_size + 1) * room_col
        row_offset = (self.room_size + 1) * room_row

        row = room_coord_row + row_offset
        col = room_coord_col + col_offset
        return (row, col)

    def _convert_index_to_float_coord(self, state_index):
        yy, xx = self.index2coord[state_index]

        # centering
        xx = xx + 0.5
        yy = yy + 0.5
        # map to [0, 1]
        xx = xx / self.ncols
        yy = yy / self.nrows
        return np.array([xx, yy])

    def reset(self, seed=None, options=None):
        self.state, info = GridWorld.reset(self, seed=seed, options=options)
        state_to_return = self.state
        if self.array_observation:
            state_to_return = self._convert_index_to_float_coord(self.state)
        return state_to_return, info

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action!"

        # save state for rendering
        if self.is_render_enabled():
            self.append_state_for_rendering(self.state)

        # take step
        next_state, reward, terminated, truncated, info = self.sample(
            self.state, action
        )
        self.state = next_state

        state_to_return = self.state
        if self.array_observation:
            state_to_return = self._convert_index_to_float_coord(self.state)

        return state_to_return, reward, terminated, truncated, info

    def get_background(self):
        """
        Returne a scene (list of shapes) representing the background
        """
        bg = Scene()

        # traps
        for y, x in self.traps:
            shape = GeometricPrimitive("POLYGON")
            shape.set_color((0.5, 0.0, 0.0))
            shape.add_vertex((x, y))
            shape.add_vertex((x + 1, y))
            shape.add_vertex((x + 1, y + 1))
            shape.add_vertex((x, y + 1))
            bg.add_shape(shape)

        # walls
        for wall in self.walls:
            y, x = wall
            shape = GeometricPrimitive("POLYGON")
            shape.set_color((0.25, 0.25, 0.25))
            shape.add_vertex((x, y))
            shape.add_vertex((x + 1, y))
            shape.add_vertex((x + 1, y + 1))
            shape.add_vertex((x, y + 1))
            bg.add_shape(shape)

        # rewards
        for y, x in self.reward_at:
            flag = GeometricPrimitive("POLYGON")
            rwd = self.reward_at[(y, x)]
            if rwd == 1.0:
                flag.set_color((0.0, 0.5, 0.0))
            elif rwd == 0.1:
                flag.set_color((0.0, 0.0, 0.5))
            else:
                flag.set_color((0.5, 0.0, 0.0))

            x += 0.5
            y += 0.25
            flag.add_vertex((x, y))
            flag.add_vertex((x + 0.25, y + 0.5))
            flag.add_vertex((x - 0.25, y + 0.5))
            bg.add_shape(flag)

        return bg

# define the BPI-VI agent
class AdaptiveAgent(AgentWithSimplePolicy):
    name="AdaptiveAgent"
    def __init__(
            self,
            env,
            gamma=1,
            horizon=100,
            delta=0.1,
            varepsilon=0.1,
            cb=0.001,
            **kwargs
    ):
        # init base class
        AgentWithSimplePolicy.__init__(self,env=env,**kwargs)

        self.gamma=gamma
        self.horizon=horizon
        self.varepsilon=varepsilon
        self.delta=delta
        self.cb=cb
        self.initial_state,_=self.env.reset()
        self.S=self.env.observation_space.n
        self.A=self.env.action_space.n
        self.bonus=np.ones((self.S,self.A))*self.horizon #should multiply H
        self.P_hat=np.ones((self.S,self.A,self.S))*1.0/self.S #estimated transition kernel
        self.Nsas=np.zeros((self.S,self.A,self.S)) #visitation count of (s,a,s')
        self.Nsa=np.zeros((self.S,self.A)) #visitation count of (s,a)
        self.R_hat=np.zeros((self.S,self.A))
        #upper and lower confidence bound
        self.q_max=np.zeros((self.horizon,self.S,self.A))
        self.q_min=np.zeros((self.horizon,self.S,self.A))
        self.v_max=np.zeros((self.horizon,self.S))
        self.v_min=np.zeros((self.horizon,self.S))
        # self.v=np.zeros((self.horizon,self.S,self.A))
        #optimality gap upper bound
        self.G=np.ones((self.horizon,self.S,self.A)) #should multiply H

        
    def fit(self,budget,**kwargs):
        """
        The parameter budget can represent the number of steps, the number of episodes etc,
        depending on the agent.
        * Interact with the environment (self.env);
        * Train the agent
        * Return useful information
        """
        T=budget
        rewards=np.zeros(T//100)
        for t in range(T):
            if ((t+1)%100==0):
                reward=self.eval()
                rewards[(t+1)//100-1]=reward
            # if ((t+1)%100==0):
            #     print("Episode",t+1)
            self.update_value()
            observation,info=self.env.reset()
            done=False
            step=0
            reward=0
            current_state=0
            while step<self.horizon:
                #terminal state is an absorbing state
                if done:
                    action=self.policy(current_state,step)
                    next_step=current_state
                    self.update(current_state,action,next_step,reward)
                else:
                    action=self.policy(observation,step)
                    next_step,reward,terminated,truncated,info=self.env.step(action)
                    #update visitation count and policy
                    self.update(observation,action,next_step,reward)
                    current_state=observation
                    observation=next_step
                    done=terminated or truncated
                step+=1
            if self.stop():
                stop_time=t+1
                print(stop_time)
                break
        return rewards
        
    def eval(self,n_simulations=100, gamma=1.0):
        """
        Monte-Carlo policy evaluation [1]_ method to estimate the mean discounted reward
        using the current policy on the evaluation environment.

        Parameters
        ----------
        eval_horizon : int, optional, default: 10**5
            Maximum episode length, representing the horizon for each simulation.
        n_simulations : int, optional, default: 10
            Number of Monte Carlo simulations to perform for the evaluation.
        gamma : float, optional, default: 1.0
            Discount factor for future rewards.

        Returns
        -------
        float
            The mean value over 'n_simulations' of the sum of rewards obtained in each simulation.

        References
        ----------
        .. [1] Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction.
            MIT Press.
        """

        episode_rewards = np.zeros(n_simulations)
        for sim in range(n_simulations):
            observation, info = self.env.reset()
            h = 0
            while h < self.horizon:
                action = self.policy(observation,h)
                observation, reward, terminated, truncated, info = self.env.step(
                    action
                )
                done = terminated or truncated
                episode_rewards[sim] += reward * np.power(gamma, h)
                h += 1
                if done:
                    break
        return episode_rewards.mean()
    
    #calculate bonus
    def beta(self,n):
        beta = np.log(3*self.S*self.A*self.horizon/self.delta) + self.S*np.log(8*np.exp(1)*(n+1))
        return beta
    
    # update counts
    def update(self,s,a,next_state,r):
        self.Nsas[s,a,next_state]+=1
        self.Nsa[s,a]+=1

        n_sa=self.Nsa[s,a]
        n_sas=self.Nsas[s,a,:]
        self.P_hat[s,a,:]=n_sas/n_sa
        self.bonus[s,a]=self.beta(n_sa)/n_sa

        self.R_hat[s,a]=r
    
    # compute empirical variance
    def emvar(self,v,s,a):
        mean=np.dot(v,self.P_hat[s,a,:])
        var=var = np.dot(self.P_hat[s, a, :], (v - mean) ** 2)
        return var
    
    # compute \hat{p}\pi*G
    def backG(self,s,a,h):
        out=0
        for i in range(self.S):
            out+=self.P_hat[s,a,i]*self.G[h+1,i,self.policy(i,h+1)]
        return out

    # update v_max v_min G                  
    def update_value(self):
        for h in range(self.horizon-1,-1,-1):
            if h==self.horizon-1:
                for s in range(self.S):
                    for a in range(self.A):
                        self.q_max[h,s,a]=min(self.horizon,self.R_hat[s,a]+self.cb*14*self.horizon**2*self.bonus[s,a])
                        self.q_min[h,s,a]=max(0,self.R_hat[s,a]-self.cb*14*self.horizon**2*self.bonus[s,a])
                        self.G[h,s,a]=min(self.horizon,self.cb*35*self.horizon**2*self.bonus[s,a])
                    self.v_max[h,s]=np.max(self.q_max[h,s,:])
                    self.v_min[h,s]=np.max(self.q_min[h,s,:])
            else:
                for s in range(self.S):
                    if (sum([1-self.P_hat[s,a,s] for a in range(self.A)])<=0.01):
                        for a in range(self.A):
                            emvarV=self.emvar(self.v_max[h+1,:],s,a)
                            self.q_max[h,s,a]=min(self.horizon,self.R_hat[s,a]+self.cb*(3*np.sqrt(emvarV*self.bonus[s,a])+14*self.horizon**2*self.bonus[s,a]))
                            self.q_min[h,s,a]=max(0,self.R_hat[s,a]-self.cb*(3*np.sqrt(emvarV*self.bonus[s,a])-14*self.horizon**2*self.bonus[s,a]))
                            self.G[h,s,a]=min(self.horizon,self.cb*(6*np.sqrt(emvarV*self.bonus[s,a])+35*self.horizon**2*self.bonus[s,a]))
                    else:
                        for a in range(self.A):
                            tmp1=np.dot(self.P_hat[s,a,:],self.v_max[h+1,:])
                            tmp2=np.dot(self.P_hat[s,a,:],self.v_min[h+1,:])
                            emvarV=self.emvar(self.v_max[h+1,:],s,a)
                            self.q_max[h,s,a]=min(self.horizon,self.R_hat[s,a]+self.cb*(3*np.sqrt(emvarV*self.bonus[s,a])+14*self.horizon**2*self.bonus[s,a]+(tmp1-tmp2)/self.horizon)+tmp1)
                            self.q_min[h,s,a]=max(0,self.R_hat[s,a]-self.cb*(3*np.sqrt(emvarV*self.bonus[s,a])-14*self.horizon**2*self.bonus[s,a]-(tmp1-tmp2)/self.horizon)+tmp2)
                            self.G[h,s,a]=min(self.horizon,self.cb*(6*np.sqrt(emvarV*self.bonus[s,a])+35*self.horizon**2*self.bonus[s,a])+(1+3/self.horizon)*self.backG(s,a,h))
                    self.v_max[h,s]=np.max(self.q_max[h,s,:])
                    self.v_min[h,s]=np.max(self.q_min[h,s,:])

    # define the agent's policy
    def policy(self,observation,h):
        return self.q_max[h,observation,:].argmax()
    
    # define stopping rule
    def stop(self):
        if self.G[0,self.initial_state,self.policy(self.initial_state,0)]<self.varepsilon/2:
            return True
        else:
            return False
    

def fit_online(num):
    print("fitting",num,"online agent")
    env=NRoom(
        nrooms=1,
        remove_walls=False,
        room_size=4,
        initial_state_distribution="center", 
        include_traps=True,
    )
    
    horizon=20 # horizon
    T=200000 # number of episodes to play
    agent=AdaptiveAgent(env=env,horizon=horizon,cb=0.002)
    reward=agent.fit(budget=T)
    return reward

def main():
    rewards=[]
    agents=range(5) # fit N agent
    with multiprocessing.Pool(processes=5) as pool:
       rewards=pool.map(fit_online,agents)
    reward_online=np.array(rewards)
    np.save("../data/onlinereward",reward_online)

if __name__ == "__main__":
    main()
