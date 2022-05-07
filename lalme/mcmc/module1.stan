data {
  int<lower=0> n_items;
  int<lower=0> n_forms_total;
  int<lower=0> n_forms[n_items];
  int<lower=0> n_anchor;
  int<lower=0> n_floating;
  int<lower=0> n_inducing;
  int<lower=0> n_locs; // number of inducing and floating points
  int<lower=0, upper=1> y[n_locs, n_forms_total]; //data, with first n_anchor rows corresponding to the anchor points
  row_vector[2] inducing_locs[n_inducing];
  row_vector[2] anchor_locs[n_anchor];
  int<lower=0> Q; // number of Gaussian processes
  real eta_smi;
  real<lower=0> gp_magnitude;
  real<lower=0> gp_length_scale;
}

transformed data {
  real delta = 1e-9;
  vector[n_locs] eta_vect = append_row(rep_vector(1, n_anchor), rep_vector(eta_smi, n_floating));
  
  matrix[n_inducing, n_inducing] K_ind;
  matrix[n_inducing, n_inducing] L_K_ind;
  matrix[n_inducing, n_anchor] K_ind_anc;
  
  K_ind = cov_exp_quad(inducing_locs, gp_magnitude, gp_length_scale) + diag_matrix(rep_vector(delta, n_inducing));
  L_K_ind = cholesky_decompose(K_ind);
  K_ind_anc = cov_exp_quad(inducing_locs, anchor_locs, gp_magnitude, gp_length_scale);
}

parameters {
  real<lower=0> mu[n_items];
  real<lower=0, upper=1> zeta[n_items];
  matrix<lower=0>[n_forms_total, Q] W;
  vector[n_forms_total - n_items] a;
  matrix[n_inducing, Q] std_normal_ind;
  real<lower=0,upper=1> x_floating[n_floating];
  real<lower=0,upper=0.9> y_floating[n_floating];
}

transformed parameters {
  matrix[n_forms_total,n_locs] phi;
  matrix[n_inducing, Q] gaussf_ind;
  matrix[n_anchor, Q] gaussf_anc;
  matrix[n_floating, Q] gaussf_flt;
  matrix[n_locs, Q] gaussf;
  vector[n_forms_total] a_complete;
  row_vector[2] floating_locs[n_floating];
  
  floating_locs[,1] = x_floating;
  floating_locs[,2] = y_floating;
  {
    int pos = 1;
    matrix[n_inducing, n_floating] K_ind_flt;
    K_ind_flt = cov_exp_quad(inducing_locs, floating_locs, gp_magnitude, gp_length_scale);
    
    for (q in 1:Q){
      vector[n_inducing] K_ind_div_Y_ind;
      
      gaussf_ind[,q] = L_K_ind * std_normal_ind[,q];
      
      K_ind_div_Y_ind = mdivide_left_tri_low(L_K_ind, gaussf_ind[,q]);
      K_ind_div_Y_ind = mdivide_right_tri_low(K_ind_div_Y_ind', L_K_ind)';
      
      gaussf_anc[,q] = K_ind_anc' * K_ind_div_Y_ind;
      gaussf_flt[,q] = K_ind_flt' * K_ind_div_Y_ind;
    }
    
    gaussf[1:n_anchor,] = gaussf_anc;
    gaussf[(n_anchor+1):n_locs,] = gaussf_flt;
    
    for (i in 1:n_items){
      a_complete[pos] = 0;
      a_complete[(pos+1):(pos+n_forms[i]-1)] = a[(pos+1-i):(pos+n_forms[i]-i-1)];
      for (l in 1:n_locs){
        phi[pos:(pos+n_forms[i]-1),l] = softmax(- a_complete[pos:(pos+n_forms[i]-1)] - W[pos:(pos+n_forms[i]-1),]* gaussf[l,]');
        //phi[pos:(pos+n_forms[i]-1),l] = softmax(- W_complete[pos:(pos+n_forms[i]-1),]* gaussf[l,]'); // remove transpose?
      }
      pos = pos + n_forms[i];
    }
  }
}

model {
  int pos = 1;
  a ~ normal(0,1);
  for (i in 1:n_inducing){
    std_normal_ind[i,] ~ normal(0,1);
  }
  to_vector(W) ~ double_exponential(0,0.1);
  
  for (l in 1:n_floating){
    floating_locs[l,1] ~ uniform(0,1);
    floating_locs[l,2] ~ uniform(0,0.9);
  }
  
  mu ~ gamma(2,2);
  zeta ~ beta(1,1);
  
  for (i in 1:n_items){
    for (f in pos:(pos+n_forms[i]-1)){
      for (l in 1:n_locs){
        //phi[pos:(pos+n_forms[i]-1),l] ~ dirichlet(rep_vector(1, n_forms[i])); //prior
        if (y[l,f] == 0){
          target += eta_vect[l] * log_sum_exp(bernoulli_lpmf(1 | zeta[i]),
                            bernoulli_lpmf(0 | zeta[i])
                              + bernoulli_lpmf( y[l,f] | 1 - exp(-mu[i] * phi[f,l])));
        }
        else {
          target += eta_vect[l] * bernoulli_lpmf(0 | zeta[i]) + eta_vect[l] * bernoulli_lpmf( y[l,f] | 1 - exp(-mu[i] * phi[f,l]));
        }
      }
    }
    pos = pos + n_forms[i];
  }
}

generated quantities {
  real log_lik[n_anchor, n_forms_total];
  int pos = 1;
  
  for (i in 1:n_items){
    for (f in pos:(pos+n_forms[i]-1)){
      for (l in 1:n_anchor){
        if (y[l,f] == 0){
          log_lik[l,f] = log_sum_exp(bernoulli_lpmf(1 | zeta[i]),
                            bernoulli_lpmf(0 | zeta[i]) + bernoulli_lpmf( y[l,f] | 1 - exp(-mu[i] * phi[f,l])));
        }
        else {
          log_lik[l,f] = bernoulli_lpmf(0 | zeta[i]) + bernoulli_lpmf( y[l,f] | 1 - exp(-mu[i] * phi[f,l]));
        }
      }
    }
    pos = pos + n_forms[i];
  }
}

