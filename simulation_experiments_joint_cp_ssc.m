%--------------------------------------------------------------------------
% this code runs all the simulation  experiments for the paper:
% K. Somandepalli and S. Narayanan, "Reinforcing Self-expressive Representation with Constraint Propagation for Face Clustering in Movies," ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Brighton, United Kingdom, 2019, pp. 4065-4069.
%--------------------------------------------------------------------------


addpath /home/krsna/Desktop/IDEA_clustering_constraint_prop/PROPACK_new2011
clear all;
close all;
clc;
rng('default')
rng(320)

% cvx_solver sdpt3
% cvx_quiet(true) ;
dvals = [ 10, 15, 20, 25, 30, 35, 40, 45, 8, 6]     ;
cn_vals = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4] ;
% 
% C_out           = cell( 1 , length(dvals)*length(cn_vals) ) ;
% F_out           = cell( 1 , length(dvals)*length(cn_vals) ) ;
% 
% C_out_joint     = cell( 1 , length(dvals)*length(cn_vals) ) ;
% F_out_joint     = cell( 1 , length(dvals)*length(cn_vals) ) ;
% 


% d = 5 ;
sigma = 0.1;
ind_n = 0 ;

% \% of constraints
tol = 1e-5;   maxiter = 500;
% ambient dim = feature dim
n = 50;
% for n  = [50] %nvals                       % The dimension of the space
for d = dvals
    
    num_subspaces = 10; %round(2*n/d);
    %round( 2 * n / d )  ;         % number of subspaces you want to generate

    
    dim_subspaces = randi([d, (n/2)-1], 1, num_subspaces);
%     5 * ones(1,num_subspaces) ;   % this a vector of dimensions of the subspaces
                                                  % e.g. dim_subspaces(3)=2 ;
                                                  % corresponds to the fact that the
                                                  % third subspace has two degree's
                                                  % of freedom (two base vectors in R^n)
    
        
                                                  
    Base_subspaces = cell(1, num_subspaces ) ;
    
                                                  % the i'th element of this cell is
                                                  % an n * dim_subspaces(i) matrix
                                                  % which is a base for the i'th
                                                  % subspace
    
    num_points_on_subspace  = dim_subspaces.*randi([2,4], 1, num_subspaces);
%     5*dim_subspaces(1)*ones(1,num_subspaces) ;
    
                                                  % number of points generated from
                                                  % each subspace
    
    
    N = sum( num_points_on_subspace ) ;           % total number of points

%     num_outliers = round(N/6) ;                            % number of outliers


% generating the subspaces

    for kk = 1 : num_subspaces

        Base_subspaces{kk}     = orth( randn( n , dim_subspaces(kk) ) ) ;

    end


% generating the points on the subspaces

    X0 = zeros( n , N) ;
    start_ind      = cumsum( num_points_on_subspace ) + 1 ;
    start_ind(end) = [] ;
    start_ind = [ 1 , start_ind ] ;
    end_ind   = start_ind + num_points_on_subspace - 1 ;

    for kk = 1 : num_subspaces

        temp = randn(  dim_subspaces(kk) , num_points_on_subspace(kk) ) ;
        temp = temp ./ repmat( sqrt(sum(temp.^2)) , size(temp,1) , 1 ) ;
        X0( : , start_ind(kk):end_ind(kk)  ) = Base_subspaces{kk} * temp ;

    end

    X1 = X0 ./ repmat( sqrt(sum(X0.^2)) , size(X0,1) , 1 ) ;

% % %     Add Outliers
%     outliers = randn(n,num_outliers) ;
%     outliers = outliers ./ repmat( sqrt(sum(outliers.^2)) , size(outliers,1) , 1 ) ;
%     X1 = [ X0 , outliers ] ;
%     clear X0;

% %     add noise with sigma
    N_  = randn(size(X1));
    N_  = N_ ./ repmat( sqrt(sum(N_.^2)) , size(N_,1) , 1 ) ;
    X  = X1 + sigma * N_ ;
    X  = X ./ repmat( sqrt(sum(X.^2)) , size(X,1) , 1 ) ;
    %make some room
    clear X1;
    clear N_;
%     
%     N = N+num_outliers ;

    num_inliers = sum(num_points_on_subspace);

    subspace_labels = [];
    for i=1:length(num_points_on_subspace)
        subspace_labels = [subspace_labels; i*ones(num_points_on_subspace(i),1)];
    end
%     subspace_labels = [subspace_labels; zeros(N-num_inliers,1)];
    
    % use 1 and 2 instead of -1 and 1
    full_subspace_constraints = -1*ones(N,N);
    for i=1:length(num_points_on_subspace)
        full_subspace_constraints(start_ind(i):end_ind(i),start_ind(i):end_ind(i)) = 1;
    end

    
	disp(['done generating data for d = ', num2str(d)])
	disp('running each');
	[C,err1,err2,err3] = admmOutlier_mat_func(X, true);
    
	C1 = C(1:N, :);
	clear C;
    for cn_x = cn_vals
		ind_n = ind_n + 1 ;
		[d,cn_x,ind_n]
		num_constraints = cn_x ; %0.05;
		R = num_subspaces;
		cn = round(num_constraints*R*(sum(size(X))-R)); %round( ((num_inliers^2 - num_inliers)/2) * num_constraints);
		
	    %     sample pairwise constrtiants from non-outliers only!!
		pc = ones(N, N); 
		pc = triu(pc, 1); 
		idx = find(pc>0); clear pc;
		cn_all = numel(idx);
		ind = randperm(cn_all); 
		ind = idx(ind(1:cn)); clear indc; 

	    %     take care that the indices are mapped to inliers only - not outliers
		Y_incomplete = sparse(zeros(N,N));   
		for kc = 1:cn
		    i = mod(ind(kc)-1,N)+1;
		    j = (ind(kc)-i)/N+1;
		    Y_incomplete(i,j) = full_subspace_constraints(i,j);
		    Y_incomplete(j,i) = full_subspace_constraints(j,i);
		end;
		disp('running joint');
        
        Data = [X; Y_incomplete];
        [WF, E, ~] = inexact_alm_mc(Data, tol, maxiter);
        F = WF(size(X,1)+1:end, :);
        E = E(size(X,1)+1:end, :);
	%         Data = [X; Y];
	%         [WF, E, ~] = inexact_alm_mc(Data, tol, maxiter);
        for l=[1e-3,1e-2,1e-1,1,10,100,1000]
            l
%             [C,WF, err1J, err2J, err3J, svp_all] = admmOutlier_nuc_norm(X, Y_incomplete, l);
            [C,WF, err1J, err2J, err3J] = cp_ssc(X, Y_incomplete, false, l);

            F_joint = WF(size(X,1)+1:end, :);
            C_joint = C(1:N, :);
% 
%             save(['expt_lambda_d',num2str(d),'_cn', num2str(cn_x) ,'_l',num2str(l),'.mat'], 'dim_subspaces', ...
%                 'num_points_on_subspace', 'subspace_labels', 'full_subspace_constraints',...
%                 'X', 'F_joint', 'C_joint', 'err1J', 'err2J', 'err3J', 'err1',...
%                  'err2', 'err3', 'C1', 'F', 'Y_incomplete', 'l')
        end
    end
end
