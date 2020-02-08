use std::collections::HashMap;

pub trait State {}
pub trait Action {}

pub trait MDP<S : State, A : Action> {

  fn nb_states(&self) -> usize;
  fn nb_actions(&self) -> usize;
  fn discount(&self) -> f64;

  // Returns for a given state action pair (their indexes) the vectors
  // corresponding to the distribution of pairs (reward,next_state), where
  // the first element of the triple is the probability of the pair.
  fn dynamics(&self, index_state : usize, index_action : usize) -> Vec<(f64,f64,usize)>;
}

// Tabular MDP
pub struct TabMDP<'a, S : State, A : Action> {
  pub states : &'a [S],
  pub actions : &'a [A],
  pub discount : f64,
  // Link a tuple (index state,index action) to the list of
  // triples (proba, reward, index next state),
  // where the proba is the numerator of the fraction
  // giving the probability of the state, where the denominator
  // is the number of states.
  pub dynamics : HashMap<(usize,usize),&'a [(usize,isize,usize)]>,
}

pub mod grid_world {

  use super::{State,Action,MDP};

  #[derive(PartialEq)]
  pub struct GridState {
    pub row : usize,
    pub col : usize,
  }

  impl State for GridState {}

  pub enum GridAction {
    Up,
    Down,
    Left,
    Right,
  }

  impl Action for GridAction {}

  pub struct GridWorld {
    nb_rows : usize,
    nb_cols : usize,
    start : GridState,
    end : GridState,
    discount: f64,
  }

  impl GridWorld {

    pub fn new(nb_rows: usize,
           nb_cols : usize,
           start : GridState,
           end : GridState,
           discount : f64) -> Self {

      GridWorld{ nb_rows,nb_cols,start,end,discount }
    }

    fn from_index(&self, index : usize) -> Option<GridState> {
      if index >= self.nb_rows*self.nb_cols {
        None
      } else {
        Some(GridState { row : index/self.nb_rows, col : index%self.nb_rows})
      }
    }

    fn to_index(&self, cell : GridState) -> Option<usize> {
      let index = cell.row*self.nb_cols + cell.col;
      if cell.col >= self.nb_cols || index >= self.nb_rows*self.nb_cols {
        None
      } else {
        Some(index)
      }
    }
  }

  impl MDP<GridState,GridAction> for GridWorld {

    fn nb_states(&self) -> usize {
      self.nb_rows*self.nb_cols
    }
    fn nb_actions(&self) -> usize {
      4
    }
    fn discount(&self) -> f64 {
      self.discount
    }

    fn dynamics(&self, index_state : usize, index_action : usize) -> Vec<(f64,f64,usize)> {
      let cell : GridState = self.from_index(index_state).unwrap();
      if cell == self.end {
        vec![(1.0,0.0,index_state)]
      } else {
        let mut next_cell = GridState {row : cell.row, col : cell.col};
        match index_action {
          0 => {next_cell.row +=1;},
          1 => {if next_cell.row > 0 {next_cell.row -=1;}},
          2 => {if next_cell.col > 0 {next_cell.col -=1;}},
          3 => {next_cell.col +=1;},
          _ => {panic!("Index action of gridworld out of bounds");}
        }
        match self.to_index(next_cell) {
          None => vec![(1.0,-1.0,index_state)],
          Some(index_next) => vec![(1.0,-1.0,index_next)],
        }
      }
    }
  }
}
