% Testing MC-RITV
% In this modification, subsampled
%k-space is provided as input.
function out = solver_MC_P_NRITV2 (maxit,mask,sense,x0,param,b)

% Input data:
% maxit: Number of iterations (e.g. 100)
% b: Undersampled k-sapce 
% mask: Undersampling pattern
% x0: Ground truth.

%% Setting parameters
global P
%  if ~exist('sigma','var')
%    sigma = 0;
%  end

mu = param.mu ;
delta = param.delta ;
beta = param.beta ;
theta = param.theta ;
tau = param.tau ;
%eta = param.eta ;
lambda_inv = param.lambda_inv ;
%sigma_bm3d = param.sigma_bm3d;
[m,n,N] = size(x0);
P = size(sense,3);

rng('default') 

%% Zero-filling solution
% Trying to move this part out of the function
% b = zeros(m,n,P,N);
% for c=1:N
%   for  p=1:P
%   b(:,:,p,c) = mask.*fft2c(sense(:,:,p).*(x0(:,:,c) + sigma*randn(m,n))) ;
%   end
% end

for c = 1:N
    zf(:,:,c) = R_adj(b(:,:,:,c),mask,sense);
end

% for ii=1:N
% figure, imshow(abs(zf(:,:,ii)));
% end
%% Defining operators

mysnr = @(x,x0) 20*log10(norm(x0,'fro')/norm(x-x0,'fro')); 
% LoG_filter = fspecial('log',[15 15], 1.5);
% LoG0 = imfilter(x0, LoG_filter,'symmetric', 'conv'); normLoG0 = norm(LoG0,'fro');
% HFEN = @(u) norm(imfilter(u, LoG_filter, 'symmetric', 'conv') - LoG0,'fro')/normLoG0;
D = @(x) cat(3,[diff(x,1,1);zeros(1,size(x,2))],[diff(x,1,2) zeros(size(x,1),1)]);
DT = @(u) -[u(1,:,1);diff(u(:,:,1),1,1)]-[u(:,1,2) diff(u(:,:,2),1,2)];
prox_alphatauG= @(t,sigma) t-bsxfun(@rdivide, t,...
max(sqrt(sum(t.^2,3))/(sigma),1));
prox2 = @(r,b,sigma) (r-sigma*b)/(sigma+1);

%% Initialization

u = zf;   
v = zeros(m,n,2,N,4);
r = zeros(m,n,P,N);
h = zeros(m,n,2,N);
KTy_u_N = zeros(m,n,N);
tmp = zeros(m,n,2,4,N);
Lh_new = zeros(size(tmp));
Lh = zeros(m,n,2,N,4);
RT_new = zeros(m,n,N);
DTh_new = zeros(m,n,N);
SNR_proposed = zeros(N,maxit);
SSIM_proposed = zeros(N,maxit);
HFEN_proposed = zeros(maxit,1);

%% Main iterations
tic;
for mm = 1:maxit
    
    %u update
    u_old = u;
    for c = 1:N
      KTy_u_N(:,:,c) =  -DT(h(:,:,:,c)) + R_adj(r(:,:,:,c),mask,sense);
    end
    u = u - tau*(KTy_u_N);
%      KTy_u = -DT(h) + R_adj(r); 
%      u = u - tau*(KTy_u);
% for ii = 1:N
%      [~,uu] = BM3D_MRI_denoise2(eta*tau,1,u(:,:,ii),sigma_bm3d(mm));  % BM3D denoising
%       u(:,:,ii) = uu;
% end
     for ii =  1:N % originally it comprised only the real part, but I added the next line to account for complex images as well. Not much difference.
    u(:,:,ii) = max(0,real(u(:,:,ii)))+...
        1i*max(imag(u(:,:,ii)),0)  ;
    end
    %v update
    v_old = v;
    for c = 1:N
        tmp(:,:,:,:,c) = L(h(:,:,:,c));
    end
    for s=1:4
        for c=1:N
            Lh(:,:,:,c,s)= reshape(tmp(:,:,:,s,c),[m n 2]);
        end
    end
    for s=1:4
        v(:,:,:,:,s) = prox_1nuc( v(:,:,:,:,s)-tau*Lh(:,:,:,:,s), tau*lambda_inv);
    end
%    KTy_v = L(h);
%    v = prox_alphatauG(v-tau*(KTy_v),tau/lambda_inv);
    tau_old = tau;
    tau = tau_old*sqrt(1+theta);
    r_old = r;
    h_old = h;
    while 1
        theta= tau/tau_old;
        ubar = u + theta*(u - u_old);
        vbar = v + theta*(v - v_old);
        betatau = beta*tau;
        for c = 1:N
              tmp2 = R(ubar(:,:,c),mask,sense);
            for p = 1:P
      
        r(:,:,p,c) = prox2(r_old(:,:,p,c) + betatau*tmp2(:,:,p),b(:,:,p,c),betatau);
            end
        end
        for c=1:N
        vv = reshape(vbar(:,:,:,c,:),[m n 2 4]);
        h(:,:,:,c) = h_old(:,:,:,c) + betatau*(LT(vv) - D(ubar(:,:,c)));
        end
        %%%%% Setting the stopping criterion
        for c=1:N
        DTh_new(:,:,c) = DT(h(:,:,:,c));
        end  
       
        for c=1:N
        RT_new(:,:,c) = R_adj(r(:,:,:,c),mask,sense);
        end
        
        for c=1:N
        Lh_new(:,:,:,:,c) = L(h(:,:,:,c));
        end
        
    if sqrt(beta)*tau*normX(-DTh_new + RT_new ...
            -KTy_u_N,Lh_new-tmp) <= delta*normY(h-h_old,r-r_old)
      break;
    else
        tau = tau*mu;
    end
    end
    for ii = 1:N
    SNR_proposed(ii,mm)=mysnr( abs(u(:,:,ii)), abs (x0(:,:,ii)));    
    SSIM_proposed(ii,mm)= ssim(abs(u(:,:,ii)), abs (x0(:,:,ii)));  
    fprintf('iteration= %d   Contrast= %d SNR= %.2f  SSIM= %.2f\n',...
    mm,ii,SNR_proposed(ii,mm),SSIM_proposed(ii,mm));
    end
    fprintf('\n');
    end
runtime = toc;


%% Outputs
out.u = u; %complex result
out.sol = abs(u); %only magnitude
out.IterationsCount = mm;
out.SamplingRate = numel(find(mask))/numel(mask);
out.Runtime = runtime;
out.SNR = SNR_proposed;
out.SSIM = SSIM_proposed;
% out.HFEN = HFEN_proposed;
% out.SNR0 = mysnr(abs(zf),abs(x0))*ones(maxit,1);
% out.SSIM0 = ssim(abs(zf),abs(x0))*ones(maxit,1);
% out.HFEN0 = HFEN(zf)*ones(maxit,1);


function t = L(u) 
	t=zeros([size(u) 4]);
	t(:,:,1,1)=u(:,:,1); 
	t(1:end-1,2:end,2,1)=(u(2:end,1:end-1,2)+u(1:end-1,1:end-1,2)+...
		u(2:end,2:end,2)+u(1:end-1,2:end,2))/4;
	t(1:end-1,1,2,1)=(u(1:end-1,1,2)+u(2:end,1,2))/4;
	t(:,:,2,2)=u(:,:,2);
	t(2:end,1:end-1,1,2)=(u(2:end,1:end-1,1)+u(1:end-1,1:end-1,1)+...
		u(2:end,2:end,1)+u(1:end-1,2:end,1))/4;
	t(1,1:end-1,1,2)=(u(1,1:end-1,1)+u(1,2:end,1))/4;
	t(2:end,:,1,3) = (u(2:end,:,1)+u(1:end-1,:,1))/2;
	t(1,:,1,3) = u(1,:,1)/2;
	t(:,2:end,2,3) = (u(:,2:end,2)+u(:,1:end-1,2))/2;
	t(:,1,2,3) = u(:,1,2)/2;
	t(1:end-1,1:end-1,1,4) = (u(1:end-1,1:end-1,1)+u(1:end-1,2:end,1))/2;
	t(1:end-1,1:end-1,2,4) = (u(1:end-1,1:end-1,2)+u(2:end,1:end-1,2))/2;

function u = LT(t)
	[height,width,d,c]=size(t);
	u=zeros(height,width,2);
	u(1:end-1,2:end,1)=t(1:end-1,2:end,1,1)+(t(1:end-1,2:end,1,2)+...
		t(1:end-1,1:end-1,1,2)+t(2:end,2:end,1,2)+t(2:end,1:end-1,1,2))/4+...
		(t(1:end-1,2:end,1,3)+t(2:end,2:end,1,3))/2+...
		(t(1:end-1,1:end-1,1,4)+t(1:end-1,2:end,1,4))/2;
	u(1:end-1,1,1)=t(1:end-1,1,1,1)+(t(1:end-1,1,1,2)+t(2:end,1,1,2))/4+...
		(t(1:end-1,1,1,3)+t(2:end,1,1,3))/2+t(1:end-1,1,1,4)/2;
	u(2:end,1:end-1,2)=t(2:end,1:end-1,2,2)+(t(2:end,1:end-1,2,1)+...
		t(1:end-1,1:end-1,2,1)+t(2:end,2:end,2,1)+t(1:end-1,2:end,2,1))/4+...
		(t(2:end,1:end-1,2,3)+t(2:end,2:end,2,3))/2+...
		(t(1:end-1,1:end-1,2,4)+t(2:end,1:end-1,2,4))/2;
	u(1,1:end-1,2)=t(1,1:end-1,2,2)+(t(1,1:end-1,2,1)+t(1,2:end,2,1))/4+...
		(t(1,1:end-1,2,3)+t(1,2:end,2,3))/2+t(1,1:end-1,2,4)/2;
    
    function a = normX(u,v)
        u = abs(u(:)); v = abs(v(:));
        a = sqrt(sum(u.^2) + sum(v.^2));
        
        
    function a = normY(h,r)
    h = abs(h(:)); r = abs(r(:));
    a = sqrt(sum(r.^2) + sum(h.^2));
         
      
     function out = prox_1nuc(v,lambda)
         [m,n,g,N]=size(v);
         out = zeros(m,n,g,N);
         for i=1:m
             for j=1:n
               out(i,j,:,:) = reshape( (prox_nuc_Erfan( (reshape(v(i,j,:,:),[2 N]))',lambda))',[1 1 2 N]);
             end
         end
         
         function out = prox_nuc_Erfan(y,lambda)
%            Eps = 1e-5;
           [U,S,V] = svd(y);
%            for i = 1:min(size(S))
%            w = lambda/(abs(S(i,i)) + Eps); 
%            S(i,i) = wthresh(S(i,i),'s',w/2 );
%            end
            S = wthresh(S,'s',lambda);
            out = U*S*V'; 
 
            
  function out = R(x,mask,sense) 
      global P
      out = zeros(size(sense));
      for i = 1:P
  out(:,:,i) = mask.*fft2c(sense(:,:,i).*x);
      end
      
 function out = R_adj (rr,mask,sense)
    global P
    out = zeros(size(mask));
    for i=1:P
    out = out + conj(sense(:,:,i)).*ifft2c(mask.*rr(:,:,i));
    end
% function x = prox_nuc(y, lambda)
%      
%      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % proximity operator of the nuclear norm
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % nucnorm = @(x) lambda*sum(svd(x)),
% % where lambda>0,
% % applied at the matrix y of size N x 2.
% % This is soft-thresholding of the singular values.
% 
% 	s = sum(y.^2,1);
% 	theta = atan2(2*dot(y(:,1),y(:,2)),s(1)-s(2))/2;
% 	c = cos(theta);
% 	s = sin(theta);
% 	v = [c -s; s c];
% 	x = y * v;
% 	tmp = max(sqrt(sum(x.^2,1)), lambda);
% 	x = bsxfun(@times, x, (tmp-lambda)./tmp) * v';
% 
% % see http://scipp.ucsc.edu/~haber/ph116A/diag2x2_11.pdf
%     
%          
%    