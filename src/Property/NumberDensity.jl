using Printf
using GaussLegendrePolynomial

function mapping(t)
  x = (1-t)/t
  return x
end

function NumberDensity_T0Mu(mu,mass)
  LambdaR_pv = 1e-3;
  LambdaV_pv = 1e3;

  LambdaR_p4 = 1e-2;
  LambdaV_p4 = 1e2;

  N_q4 = 400;
  q4Exp,wq4=GauLeg_Log(LambdaR_p4, LambdaV_p4,N_q4);

  #t,wt=GauLeg(0.0,pi/2,N_q4)
  t,wt=GauLeg(0.0,1.0,N_q4)


  N_qv = 2000;
  qvExp,wqv=GauLeg_Log(LambdaR_pv, LambdaV_pv, N_qv);
  #tp,wtp=GauLeg(0.0,1.0,N_qv)
  

  number_density=0.0;
  f=open("qv_fi.txt","w")
  for i=N_qv:N_qv
    q= 1000;
    #q= qvExp[i];
    #q= (1-tp[i])/tp[i];
    fi=0;
    for n=1:N_q4
      #q4=q4Exp[n];
      q4=(1-t[n])/t[n];
      #q4=tan(t[n]);
      A=1
      B=mass
      C=1
      _q4=Complex(q4,mu)
      #kernel =2.0* real(4.0*im*_q4*C/(q*q*A*A+_q4*_q4*C*C+B*B));
      kernel =2.0* real(4.0*im*_q4*C/(q*q*A*A+_q4*_q4*C*C+B*B))/t[n]^2;

      #fi  +=1/(4*pi) * kernel *  wq4[n] ;
      fi  +=1/(4*pi) * kernel *  wt[n];
      #fi  +=1/(4*pi) * kernel *  wt[n]*sec(t[n])^2 ;
      write(f,@sprintf("%le	%le\n",q4,kernel));
    end
    println(fi)
    number_density += 3/(pi*pi)*q*q*wqv[i]*fi;
    #number_density += 3/(pi*pi)*q*q*wtp[i]/tp[i]^2*fi;

    #write(f,@sprintf("%le	%le\n",q,fi));
  end
  close(f)
  return number_density;
end #function NumberDensity_T0Mu


NumberDensity_T0Mu(0.5,0.2)