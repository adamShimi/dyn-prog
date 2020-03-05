use crate::mdp::{State, Action, ActionValue, MDP, Policy};

// Public API

pub fn run_monte_carlo_first_visit<S,A,M>(prob : &M) -> Policy
  where S : State,
        A : Action,
        M : MDP<S,A> {

  let mut pol = Policy { choice : vec![(0..prob.nb_actions()).collect::<Vec<usize>>();
                                       prob.nb_states()] };
  let mut new_pol;
  let mut action_value =
    ActionValue { value : vec![0.0; prob.nb_actions()] };
  let mut episode;

  loop {
    episode = get_episode(prob, &pol);
    new_pol = update_first_visit(&pol, &mut action_value, episode);

    if new_pol == pol {
      break;
    } else {
      std::mem::replace(&mut pol, new_pol);
    }
  }
  new_pol
}

struct Episode {
  events : Vec<(usize,usize,f64)>
}

fn get_episode<S,A,M>(prob : &M, pol : &Policy, start_index : usize) -> Episode
  where S : State,
        A : Action,
        M : MDP<S,A> {
  unimplemented!();
}

fn update_first_visit(pol : &Policy,
                      action_value: &mut ActionValue,
                      episode: Episode) -> Policy {
  unimplemented!();
}
