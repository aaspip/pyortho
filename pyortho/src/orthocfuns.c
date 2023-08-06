#include <Python.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <numpy/arrayobject.h>

/*NOTE: PS indicates PySeistr*/
#define PS_NINT(x) ((int)((x)>0.0?(x)+0.5:(x)-0.5))
#define PS_MAX_DIM 9

/*sf functions*/
typedef void (*operator)(bool,bool,int,int,float*,float*);

void ps_adjnull (bool adj /* adjoint flag */, 
		 bool add /* addition flag */, 
		 int nx   /* size of x */, 
		 int ny   /* size of y */, 
		 float* x, 
		 float* y) 
/*< Zeros out the output (unless add is true). 
  Useful first step for any linear operator. >*/
{
    int i;
    
    if(add) return;
    
    if(adj) {
	for (i = 0; i < nx; i++) {
	    x[i] = 0.;
	}
    } else {
	for (i = 0; i < ny; i++) {
	    y[i] = 0.;
	}
    }
}


void *ps_alloc (size_t n    /* number of elements */, 
			  size_t size /* size of one element */)
	  /*< output-checking allocation >*/
{
    void *ptr; 
    
    size *= n;
    
    ptr = malloc (size);

    if (NULL == ptr)
	{
	printf("cannot allocate %lu bytes:", size);
	return NULL;
	}

    return ptr;
}

float *ps_floatalloc (size_t n /* number of elements */)
	  /*< float allocation >*/ 
{
    float *ptr;
    ptr = (float*) ps_alloc (n,sizeof(float));
    return ptr;
}

int *ps_intalloc (size_t n /* number of elements */)
	  /*< int allocation >*/  
{
    int *ptr;
    ptr = (int*) ps_alloc (n,sizeof(int));
    return ptr;
}

/*from decart.c*/
int ps_first_index (int i          /* dimension [0...dim-1] */, 
		    int j        /* line coordinate */, 
		    int dim        /* number of dimensions */, 
		    const int *n /* box size [dim] */, 
		    const int *s /* step [dim] */)
/*< Find first index for multidimensional transforms >*/
{
    int i0, n123, ii;
    int k;

    n123 = 1;
    i0 = 0;
    for (k=0; k < dim; k++) {
	if (k == i) continue;
	ii = (j/n123)%n[k]; /* to cartesian */
	n123 *= n[k];	
	i0 += ii*s[k];      /* back to line */
    }

    return i0;
}

/*from cblas */
double ps_cblas_dsdot(int n, const float *x, int sx, const float *y, int sy)
/*< x'y float -> double >*/
{
    int i, ix, iy;
    double dot;

    dot = 0.;

    for (i=0; i < n; i++) {
	ix = i*sx;
	iy = i*sy;
        dot += (double) x[ix] * y[iy];
    }

    return dot;
}

void ps_cblas_sscal(int n, float alpha, float *x, int sx)
/*< x = alpha*x >*/
{
    int i, ix;

    for (i=0; i < n; i++) {
        ix = i*sx;
	x[ix] *= alpha;
    }
}


void ps_cblas_saxpy(int n, float a, const float *x, int sx, float *y, int sy)
/*< y += a*x >*/
{
    int i, ix, iy;

    for (i=0; i < n; i++) {
	ix = i*sx;
	iy = i*sy;
	y[iy] += a * x[ix];
    }
}

void ps_cblas_sswap(int n, float *x, int sx, float* y, int sy) 
/*< swap x and y >*/
{
    int i, ix, iy;
    float t;

    for (i=0; i < n; i++) {
	ix = i*sx;
	iy = i*sy;
	t = x[ix];
	x[ix] = y[iy];
	y[iy] = t;
    }
}

float ps_cblas_sdot(int n, const float *x, int sx, const float *y, int sy)
/*< x'y float -> complex >*/
{
    int i, ix, iy;
    float dot;

    dot = 0.;

    for (i=0; i < n; i++) {
	ix = i*sx;
	iy = i*sy;
	dot += x[ix] * y[iy];
    }

    return dot;
}


float ps_cblas_snrm2 (int n, const float* x, int sx) 
/*< sum x_i^2 >*/
{
    int i, ix;
    float xn;

    xn = 0.0;

    for (i=0; i < n; i++) {
	ix = i*sx;
	xn += x[ix]*x[ix];
    }
    return xn;
}

/*from conjugate*/
static int np, nx, nr, nd;
static float *r, *sp, *sx, *sr, *gp, *gx, *gr;
static float eps, tol;
static bool verb, hasp0;

void ps_conjgrad_init(int np1     /* preconditioned size */, 
		      int nx1     /* model size */, 
		      int nd1     /* data size */, 
		      int nr1     /* residual size */, 
		      float eps1  /* scaling */,
		      float tol1  /* tolerance */, 
		      bool verb1  /* verbosity flag */, 
		      bool hasp01 /* if has initial model */) 
/*< solver constructor >*/
{
    np = np1; 
    nx = nx1;
    nr = nr1;
    nd = nd1;
    eps = eps1*eps1;
    tol = tol1;
    verb = verb1;
    hasp0 = hasp01;

    r = ps_floatalloc(nr);  
    sp = ps_floatalloc(np);
    gp = ps_floatalloc(np);
    sx = ps_floatalloc(nx);
    gx = ps_floatalloc(nx);
    sr = ps_floatalloc(nr);
    gr = ps_floatalloc(nr);
}

void ps_conjgrad_close(void) 
/*< Free allocated space >*/
{
    free (r);
    free (sp);
    free (gp);
    free (sx);
    free (gx);
    free (sr);
    free (gr);
}

void ps_conjgrad(operator prec  /* data preconditioning */, 
		 operator oper  /* linear operator */, 
		 operator shape /* shaping operator */, 
		 float* p          /* preconditioned model */, 
		 float* x          /* estimated model */, 
		 float* dat        /* data */, 
		 int niter         /* number of iterations */) 
/*< Conjugate gradient solver with shaping >*/
{
    double gn, gnp, alpha, beta, g0, dg, r0;
    float *d=NULL;
    int i, iter;
    
    if (NULL != prec) {
	d = ps_floatalloc(nd); 
	for (i=0; i < nd; i++) {
	    d[i] = - dat[i];
	}
	prec(false,false,nd,nr,d,r);
    } else {
	for (i=0; i < nr; i++) {
	    r[i] = - dat[i];
	}
    }
    
    if (hasp0) { /* initial p */
	shape(false,false,np,nx,p,x);
	if (NULL != prec) {
	    oper(false,false,nx,nd,x,d);
	    prec(false,true,nd,nr,d,r);
	} else {
	    oper(false,true,nx,nr,x,r);
	}
    } else {
	for (i=0; i < np; i++) {
	    p[i] = 0.;
	}
	for (i=0; i < nx; i++) {
	    x[i] = 0.;
	}
    } 
    
    dg = g0 = gnp = 0.;
    r0 = ps_cblas_dsdot(nr,r,1,r,1);
    if (r0 == 0.) {
	if (verb) printf("zero residual: r0=%g \n",r0);
	return;
    }

    for (iter=0; iter < niter; iter++) {
	for (i=0; i < np; i++) {
	    gp[i] = eps*p[i];
	}
	for (i=0; i < nx; i++) {
	    gx[i] = -eps*x[i];
	}

	if (NULL != prec) {
	    prec(true,false,nd,nr,d,r);
	    oper(true,true,nx,nd,gx,d);
	} else {
	    oper(true,true,nx,nr,gx,r);
	}

	shape(true,true,np,nx,gp,gx);
	shape(false,false,np,nx,gp,gx);

	if (NULL != prec) {
	    oper(false,false,nx,nd,gx,d);
	    prec(false,false,nd,nr,d,gr);
	} else {
	    oper(false,false,nx,nr,gx,gr);
	}

	gn = ps_cblas_dsdot(np,gp,1,gp,1);

	if (iter==0) {
	    g0 = gn;

	    for (i=0; i < np; i++) {
		sp[i] = gp[i];
	    }
	    for (i=0; i < nx; i++) {
		sx[i] = gx[i];
	    }
	    for (i=0; i < nr; i++) {
		sr[i] = gr[i];
	    }
	} else {
	    alpha = gn / gnp;
	    dg = gn / g0;

	    if (alpha < tol || dg < tol) {
		if (verb) 
		    printf(
			"convergence in %d iterations, alpha=%g, gd=%g \n",
			iter,alpha,dg);
		break;
	    }

	    ps_cblas_saxpy(np,alpha,sp,1,gp,1);
	    ps_cblas_sswap(np,sp,1,gp,1);

	    ps_cblas_saxpy(nx,alpha,sx,1,gx,1);
	    ps_cblas_sswap(nx,sx,1,gx,1);

	    ps_cblas_saxpy(nr,alpha,sr,1,gr,1);
	    ps_cblas_sswap(nr,sr,1,gr,1);
	}

	beta = ps_cblas_dsdot(nr,sr,1,sr,1) + eps*(ps_cblas_dsdot(np,sp,1,sp,1) - ps_cblas_dsdot(nx,sx,1,sx,1));
	
	if (verb) printf("iteration %d res: %f grad: %f\n",
			     iter,ps_cblas_snrm2(nr,r,1)/r0,dg);

	alpha = - gn / beta;

	ps_cblas_saxpy(np,alpha,sp,1,p,1);
	ps_cblas_saxpy(nx,alpha,sx,1,x,1);
	ps_cblas_saxpy(nr,alpha,sr,1,r,1);

	gnp = gn;
    }

    if (NULL != prec) free (d);

}

/*from triangle.c*/
typedef struct ps_Triangle *ps_triangle;
/* abstract data type */

struct ps_Triangle {
    float *tmp, wt;
    int np, nb, nx;
    bool box;
};

static void fold (int o, int d, int nx, int nb, int np, 
		  const float *x, float* tmp);
static void fold2 (int o, int d, int nx, int nb, int np, 
		   float *x, const float* tmp);
static void doubint (int nx, float *x, bool der);
static void triple (int o, int d, int nx, int nb, 
		    float* x, const float* tmp, bool box, float wt);
static void triple2 (int o, int d, int nx, int nb, 
		     const float* x, float* tmp, bool box, float wt);

ps_triangle ps_triangle_init (int  nbox /* triangle length */, 
			      int  ndat /* data length */,
                              bool box  /* if box instead of triangle */)
/*< initialize >*/
{
    ps_triangle tr;

    tr = (ps_triangle) ps_alloc(1,sizeof(*tr));

    tr->nx = ndat;
    tr->nb = nbox;
    tr->box = box;
    tr->np = ndat + 2*nbox;
    
    if (box) {
	tr->wt = 1.0/(2*nbox-1);
    } else {
	tr->wt = 1.0/(nbox*nbox);
    }
    
    tr->tmp = ps_floatalloc(tr->np);

    return tr;
}

static void fold (int o, int d, int nx, int nb, int np, 
		  const float *x, float* tmp)
{
    int i, j;

    /* copy middle */
    for (i=0; i < nx; i++) 
	tmp[i+nb] = x[o+i*d];
    
    /* reflections from the right side */
    for (j=nb+nx; j < np; j += nx) {
	for (i=0; i < nx && i < np-j; i++)
	    tmp[j+i] = x[o+(nx-1-i)*d];
	j += nx;
	for (i=0; i < nx && i < np-j; i++)
	    tmp[j+i] = x[o+i*d];
    }
    
    /* reflections from the left side */
    for (j=nb; j >= 0; j -= nx) {
	for (i=0; i < nx && i < j; i++)
	    tmp[j-1-i] = x[o+i*d];
	j -= nx;
	for (i=0; i < nx && i < j; i++)
	    tmp[j-1-i] = x[o+(nx-1-i)*d];
    }
}

static void fold2 (int o, int d, int nx, int nb, int np, 
		   float *x, const float* tmp)
{
    int i, j;

    /* copy middle */
    for (i=0; i < nx; i++) 
	x[o+i*d] = tmp[i+nb];

    /* reflections from the right side */
    for (j=nb+nx; j < np; j += nx) {
	for (i=0; i < nx && i < np-j; i++)
	    x[o+(nx-1-i)*d] += tmp[j+i];
	j += nx;
	for (i=0; i < nx && i < np-j; i++)
	    x[o+i*d] += tmp[j+i];
    }
    
    /* reflections from the left side */
    for (j=nb; j >= 0; j -= nx) {
	for (i=0; i < nx && i < j; i++)
	    x[o+i*d] += tmp[j-1-i];
	j -= nx;
	for (i=0; i < nx && i < j; i++)
	    x[o+(nx-1-i)*d] += tmp[j-1-i];
    }
}
    
static void doubint (int nx, float *xx, bool der)
{
    int i;
    float t;

    /* integrate backward */
    t = 0.;
    for (i=nx-1; i >= 0; i--) {
	t += xx[i];
	xx[i] = t;
    }

    if (der) return;

    /* integrate forward */
    t=0.;
    for (i=0; i < nx; i++) {
	t += xx[i];
	xx[i] = t;
    }
}

static void doubint2 (int nx, float *xx, bool der)
{
    int i;
    float t;


    /* integrate forward */
    t=0.;
    for (i=0; i < nx; i++) {
	t += xx[i];
	xx[i] = t;
    }

    if (der) return;

    /* integrate backward */
    t = 0.;
    for (i=nx-1; i >= 0; i--) {
	t += xx[i];
	xx[i] = t;
    }
}

static void triple (int o, int d, int nx, int nb, float* x, const float* tmp, bool box, float wt)
{
    int i;
    const float *tmp1, *tmp2;
    
    if (box) {
	tmp2 = tmp + 2*nb;

	for (i=0; i < nx; i++) {
	    x[o+i*d] = (tmp[i+1] - tmp2[i])*wt;
	}
    } else {
	tmp1 = tmp + nb;
	tmp2 = tmp + 2*nb;

	for (i=0; i < nx; i++) {
	    x[o+i*d] = (2.*tmp1[i] - tmp[i] - tmp2[i])*wt;
	}
    }
}

static void dtriple (int o, int d, int nx, int nb, float* x, const float* tmp, float wt)
{
    int i;
    const float *tmp2;

    tmp2 = tmp + 2*nb;
    
    for (i=0; i < nx; i++) {
	x[o+i*d] = (tmp[i] - tmp2[i])*wt;
    }
}

static void triple2 (int o, int d, int nx, int nb, const float* x, float* tmp, bool box, float wt)
{
    int i;

    for (i=0; i < nx + 2*nb; i++) {
	tmp[i] = 0;
    }

    if (box) {
	ps_cblas_saxpy(nx,  +wt,x+o,d,tmp+1   ,1);
	ps_cblas_saxpy(nx,  -wt,x+o,d,tmp+2*nb,1);
    } else {
	ps_cblas_saxpy(nx,  -wt,x+o,d,tmp     ,1);
	ps_cblas_saxpy(nx,2.*wt,x+o,d,tmp+nb  ,1);
	ps_cblas_saxpy(nx,  -wt,x+o,d,tmp+2*nb,1);
    }
}

static void dtriple2 (int o, int d, int nx, int nb, const float* x, float* tmp, float wt)
{
    int i;

    for (i=0; i < nx + 2*nb; i++) {
	tmp[i] = 0;
    }

    ps_cblas_saxpy(nx,  wt,x+o,d,tmp     ,1);
    ps_cblas_saxpy(nx, -wt,x+o,d,tmp+2*nb,1);
}

void ps_smooth (ps_triangle tr  /* smoothing object */, 
		int o, int d    /* trace sampling */, 
		bool der        /* if derivative */, 
		float *x        /* data (smoothed in place) */)
/*< apply triangle smoothing >*/
{
    fold (o,d,tr->nx,tr->nb,tr->np,x,tr->tmp);
    doubint (tr->np,tr->tmp,(bool) (tr->box || der));
    triple (o,d,tr->nx,tr->nb,x,tr->tmp, tr->box, tr->wt);
}

void ps_dsmooth (ps_triangle tr  /* smoothing object */, 
		int o, int d    /* trace sampling */, 
		bool der        /* if derivative */, 
		float *x        /* data (smoothed in place) */)
/*< apply triangle smoothing >*/
{
    fold (o,d,tr->nx,tr->nb,tr->np,x,tr->tmp);
    doubint (tr->np,tr->tmp,(bool) (tr->box || der));
    dtriple (o,d,tr->nx,tr->nb,x,tr->tmp, tr->wt);
}

void ps_smooth2 (ps_triangle tr  /* smoothing object */, 
		 int o, int d    /* trace sampling */, 
		 bool der        /* if derivative */,
		 float *x        /* data (smoothed in place) */)
/*< apply adjoint triangle smoothing >*/
{
    triple2 (o,d,tr->nx,tr->nb,x,tr->tmp, tr->box, tr->wt);
    doubint2 (tr->np,tr->tmp,(bool) (tr->box || der));
    fold2 (o,d,tr->nx,tr->nb,tr->np,x,tr->tmp);
}

void ps_dsmooth2 (ps_triangle tr  /* smoothing object */, 
		 int o, int d    /* trace sampling */, 
		 bool der        /* if derivative */,
		 float *x        /* data (smoothed in place) */)
/*< apply adjoint triangle smoothing >*/
{
    dtriple2 (o,d,tr->nx,tr->nb,x,tr->tmp, tr->wt);
    doubint2 (tr->np,tr->tmp,(bool) (tr->box || der));
    fold2 (o,d,tr->nx,tr->nb,tr->np,x,tr->tmp);
}

void  ps_triangle_close(ps_triangle tr)
/*< free allocated storage >*/
{
    free (tr->tmp);
    free (tr);
}

/*from trianglen.c*/
static int *ntri, s[PS_MAX_DIM], nd, dim;
static ps_triangle *tr;
static float *tmp;

void ps_trianglen_init (int ndim  /* number of dimensions */, 
			int *nbox /* triangle radius [ndim] */, 
			int *ndat /* data dimensions [ndim] */)
/*< initialize >*/
{
    int i;

    dim = ndim;
    ntri = ps_intalloc(dim);

    tr = (ps_triangle*) ps_alloc(dim,sizeof(ps_triangle));

    nd = 1;
    for (i=0; i < dim; i++) {
	tr[i] = (nbox[i] > 1)? ps_triangle_init (nbox[i],ndat[i],false): NULL;
	s[i] = nd;
	ntri[i] = ndat[i];
	nd *= ndat[i];
    }
    tmp = ps_floatalloc (nd);
}

void ps_trianglen_lop (bool adj, bool add, int nx, int ny, float* x, float* y)
/*< linear operator >*/
{
    int i, j, i0;

//     if (nx != ny || nx != nd) 
// 	ps_error("%s: Wrong data dimensions: nx=%d, ny=%d, nd=%d",
// 		 __FILE__,nx,ny,nd);

    ps_adjnull (adj,add,nx,ny,x,y);
  
    if (adj) {
	for (i=0; i < nd; i++) {
	    tmp[i] = y[i];
	}
    } else {
	for (i=0; i < nd; i++) {
	    tmp[i] = x[i];
	}
    }

  
    for (i=0; i < dim; i++) {
	if (NULL != tr[i]) {
	    for (j=0; j < nd/ntri[i]; j++) {
		i0 = ps_first_index (i,j,dim,ntri,s);
		ps_smooth2 (tr[i], i0, s[i], false, tmp);
	    }
	}
    }
	
    if (adj) {
	for (i=0; i < nd; i++) {
	    x[i] += tmp[i];
	}
    } else {
	for (i=0; i < nd; i++) {
	    y[i] += tmp[i];
	}
    }    
}

void ps_trianglen_close(void)
/*< free allocated storage >*/
{
    int i;

    free (tmp);

    for (i=0; i < dim; i++) {
	if (NULL != tr[i]) ps_triangle_close (tr[i]);
    }

    free(tr);
    free(ntri);
}


/*from weight.c*/
static float* weig;

void ps_weight_init(float *w1)
/*< initialize >*/
{
    weig = w1;
}

void ps_weight_lop (bool adj, bool add, int nx, int ny, float* xx, float* yy)
/*< linear operator >*/
{
    int i;

//     if (ny!=nx) ps_error("%s: size mismatch: %d != %d",__FILE__,ny,nx);

    ps_adjnull (adj, add, nx, ny, xx, yy);
  
    if (adj) {
        for (i=0; i < nx; i++) {
	    xx[i] += yy[i] * weig[i];
	}
    } else {
        for (i=0; i < nx; i++) {
            yy[i] += xx[i] * weig[i];
	}
    }

}


/*from divn.c*/
static int niter, ndivn;
static float *p;

void ps_divn_init(int ndim   /* number of dimensions */, 
		  int nd     /* data size */, 
		  int *ndat  /* data dimensions [ndim] */, 
		  int *nbox  /* smoothing radius [ndim] */, 
		  int niter1 /* number of iterations */,
		  bool verb  /* verbosity */) 
/*< initialize >*/
{
    niter = niter1;
    ndivn = nd;

    ps_trianglen_init(ndim, nbox, ndat);
    ps_conjgrad_init(nd, nd, nd, nd, 1., 1.e-6, verb, false);
    p = ps_floatalloc (nd);
}

void ps_divn_close (void)
/*< free allocated storage >*/
{
    ps_trianglen_close();
    ps_conjgrad_close();
    free (p);
}

void ps_divn (float* num, float* den,  float* rat)
/*< smoothly divide rat=num/den >*/
{
    ps_weight_init(den);
    ps_conjgrad(NULL, ps_weight_lop,ps_trianglen_lop,p,rat,num,niter); 
}

void ps_divne (float* num, float* den,  float* rat, float eps)
/*< smoothly divide rat=num/den with preconditioning >*/
{
    int i;
    double norm;

    if (eps > 0.0f) {
	for (i=0; i < ndivn; i++) {
	    norm = 1.0/hypot(den[i],eps);

	    num[i] *= norm;
	    den[i] *= norm;
	}
    } 

    norm = ps_cblas_dsdot(ndivn,den,1,den,1);
    if (norm == 0.0) {
	for (i=0; i < ndivn; i++) {
	    rat[i] = 0.0;
	}
	return;
    }
    norm = sqrt(ndivn/norm);

    for (i=0; i < ndivn; i++) {
	num[i] *= norm;
	den[i] *= norm;
    }   

    ps_weight_init(den);
    ps_conjgrad(NULL, ps_weight_lop,ps_trianglen_lop,p,rat,num,niter); 
}

void ps_divn_combine (const float* one, const float* two, float *prod)
/*< compute product of two divisions >*/
{
    int i;
    float p;

    for (i=0; i < ndivn; i++) {
	p = sqrtf(fabsf(one[i]*two[i]));
	if ((one[i] > 0. && two[i] < 0. && -two[i] >= one[i]) ||
	    (one[i] < 0. && two[i] > 0. && two[i] >= -one[i])) 
	    p = -p;
	p += 1.;
	p *= p;
	p *= p/16.;
	prod[i] = p;	
    }
}

void mcp(float *dst /*destination*/, 
		float *src /*source*/,
		int s1d /*starting index in dst*/,
		int s1s /*starting index in src*/,
		int l1  /*copy length in axis1*/)
/*<memory copy in 1D case>*/
{
	int i1=0;
	for(i1=0;i1<l1;i1++)
	{
		dst[s1d+i1]=src[s1s+i1];
	}
}

static PyObject *Clocalortho(PyObject *self, PyObject *args){
    
	/**initialize data input**/
    int i2, n1, n2, n3, n123, nd2;
    float *data,*din;
    int verb;
    int r1,r2,r3,diff1,diff2,diff3,box1,box2,box3;
    int repeat,adj;
	ps_triangle tr;
	int ndata2;
	float *data2, *sig, *noi, *sig2, *noi2, *rat;
	int id, nd;
	float remove;
	
    PyObject *f1=NULL;
    PyObject *arrf1=NULL;
    PyObject *f2=NULL;
    PyObject *arrf2=NULL;
    
	int niter; float eps;
    
	PyArg_ParseTuple(args, "OOiiiiiiifi", &f1, &f2, &n1, &n2, &n3, &r1, &r2, &r3, &niter, &eps, &verb);
        
    printf("n1=%d,n2=%d,n3=%d,r1=%d,r2=%d,r3=%d\n",n1,n2,n3,r1,r2,r3);
    printf("niter=%d,eps=%g,verb=%d\n",niter,eps,verb);
    
	n123=n1*n2*n3;
	
    arrf1 = PyArray_FROM_OTF(f1, NPY_FLOAT, NPY_IN_ARRAY);
    arrf2 = PyArray_FROM_OTF(f2, NPY_FLOAT, NPY_IN_ARRAY);
    
    nd2=PyArray_NDIM(arrf1);
    npy_intp *sp=PyArray_SHAPE(arrf1);
	
    if (*sp != n123)
    {
    	printf("Dimension mismatch, N_input = %d, N_data = %d\n", *sp, n123);
    	return NULL;
    }

    int dim, dim1, i, j, n[PS_MAX_DIM], rect[PS_MAX_DIM], s[PS_MAX_DIM];
    int nrep, irep, i0;
    bool diff[PS_MAX_DIM], box[PS_MAX_DIM];
    
    
	if(n3>1)
		dim=3;
	else
		dim=2;
		
	if(r3>1)
	dim1=2;
	else
	{
	if(r2>1)
	dim1=1;
	else
	dim1=0;
	}
	
	n[0]=n1;n[1]=n2;n[2]=n3;
	s[0]=1;s[1]=n1;s[2]=n1*n2;
	rect[0]=r1;rect[1]=r2;rect[2]=r3;
	
	nd=n1*n2*n3;
	
	printf("dim=%d,dim1=%d\n",dim,dim1);
	printf("nd=%d\n",nd);

    noi = ps_floatalloc(nd);
    sig = ps_floatalloc(nd);
    rat = ps_floatalloc(nd);    

    noi2 = ps_floatalloc(nd);
    sig2 = ps_floatalloc(nd);

    data2 = ps_floatalloc(nd*3);
    
    /*reading data*/
    for (i=0; i<nd; i++)
    {
        sig[i]=*((float*)PyArray_GETPTR1(arrf1,i));
    }

    for (i=0; i<nd; i++)
    {
        noi[i]=*((float*)PyArray_GETPTR1(arrf2,i));
    }

    ps_divn_init(dim, nd, n, rect, niter, verb);

    for (id=0; id < nd; id++) {
	noi2[id] = noi[id];
	sig2[id] = sig[id];
    }

    ps_divne (noi, sig, rat, eps);

    for (id=0; id < nd; id++) {
	remove = rat[id]*sig2[id];
	noi2[id] -= remove;
	sig2[id] += remove;
    }
    
    mcp(data2,sig2,0,0,nd);
    mcp(data2,noi2,nd,0,nd);
    mcp(data2,rat,nd*2,0,nd);
    
    ndata2=nd*3;
    /*Below is the output part*/
    PyArrayObject *vecout;
	npy_intp dims[2];
	dims[0]=ndata2;dims[1]=1;
	/* Parse tuples separately since args will differ between C fcns */
	/* Make a new double vector of same dimension */
	vecout=(PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_FLOAT);
	for(i=0;i<dims[0];i++)
		(*((float*)PyArray_GETPTR1(vecout,i))) = data2[i];

	/*free memory*/
	free(sig);free(noi);free(sig2);free(noi2);free(rat);free(data2);
	
	return PyArray_Return(vecout);
	
}

static PyObject *Clocalsimi(PyObject *self, PyObject *args){
    
	/**initialize data input**/
    int i2, n1, n2, n3, n123, nd2;
    float *data,*din;
    int verb;
    int r1,r2,r3,diff1,diff2,diff3,box1,box2,box3;
    int repeat,adj;
	ps_triangle tr;
	float *one, *two, *rat1, *rat2;
	int id, nd;
	
    PyObject *f1=NULL;
    PyObject *arrf1=NULL;
    PyObject *f2=NULL;
    PyObject *arrf2=NULL;
    
	int niter; float eps;
    
	PyArg_ParseTuple(args, "OOiiiiiiifi", &f1, &f2, &n1, &n2, &n3, &r1, &r2, &r3, &niter, &eps, &verb);
        
    printf("n1=%d,n2=%d,n3=%d,r1=%d,r2=%d,r3=%d\n",n1,n2,n3,r1,r2,r3);
    printf("niter=%d,eps=%g,verb=%d\n",niter,eps,verb);
    
	n123=n1*n2*n3;
	
    arrf1 = PyArray_FROM_OTF(f1, NPY_FLOAT, NPY_IN_ARRAY);
    arrf2 = PyArray_FROM_OTF(f2, NPY_FLOAT, NPY_IN_ARRAY);
    
    nd2=PyArray_NDIM(arrf1);
    npy_intp *sp=PyArray_SHAPE(arrf1);
	
    if (*sp != n123)
    {
    	printf("Dimension mismatch, N_input = %d, N_data = %d\n", *sp, n123);
    	return NULL;
    }

    int dim, dim1, i, j, n[PS_MAX_DIM], rect[PS_MAX_DIM], s[PS_MAX_DIM];
    int nrep, irep, i0;
    bool diff[PS_MAX_DIM], box[PS_MAX_DIM];
    
    
	if(n3>1)
		dim=3;
	else
		dim=2;
		
	if(r3>1)
	dim1=2;
	else
	{
	if(r2>1)
	dim1=1;
	else
	dim1=0;
	}
	
	n[0]=n1;n[1]=n2;n[2]=n3;
	s[0]=1;s[1]=n1;s[2]=n1*n2;
	rect[0]=r1;rect[1]=r2;rect[2]=r3;
	nd=n1*n2*n3;
	
	printf("dim=%d,dim1=%d\n",dim,dim1);
	printf("nd=%d\n",nd);  

    ps_divn_init(dim, nd, n, rect, niter, verb);
	
    one  = ps_floatalloc(nd);
    two  = ps_floatalloc(nd);
    rat1 = ps_floatalloc(nd);
    rat2 = ps_floatalloc(nd);

    /*reading data*/
    for (i=0; i<nd; i++)
    {
        one[i]=*((float*)PyArray_GETPTR1(arrf1,i));
    }

    for (i=0; i<nd; i++)
    {
        two[i]=*((float*)PyArray_GETPTR1(arrf2,i));
    }
	
	ps_divne(one,two,rat1,eps);
    ps_divne(two,one,rat2,eps);

	/* combination */
	ps_divn_combine (rat1,rat2,rat1);


    /*Below is the output part*/
    PyArrayObject *vecout;
	npy_intp dims[2];
	dims[0]=nd;dims[1]=1;
	/* Parse tuples separately since args will differ between C fcns */
	/* Make a new double vector of same dimension */
	vecout=(PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_FLOAT);
	for(i=0;i<dims[0];i++)
		(*((float*)PyArray_GETPTR1(vecout,i))) = rat1[i];

	/*free memory*/
	free(one);free(two);free(rat1);free(rat2);
	
	return PyArray_Return(vecout);
	
}

// documentation for each functions.
static char dipcfun_document[] = "Document stuff for dip...";

// defining our functions like below:
// function_name, function, METH_VARARGS flag, function documents
static PyMethodDef functions[] = {
  {"Clocalortho", Clocalortho, METH_VARARGS, dipcfun_document},
  {"Clocalsimi", Clocalsimi, METH_VARARGS, dipcfun_document},
  {NULL, NULL, 0, NULL}
};

// initializing our module informations and settings in this structure
// for more informations, check head part of this file. there are some important links out there.
static struct PyModuleDef dipcfunModule = {
  PyModuleDef_HEAD_INIT, // head informations for Python C API. It is needed to be first member in this struct !!
  "orthocfun",  // module name
  NULL, // means that the module does not support sub-interpreters, because it has global state.
  -1,
  functions  // our functions list
};

// runs while initializing and calls module creation function.
PyMODINIT_FUNC PyInit_orthocfun(void){
  
    PyObject *module = PyModule_Create(&dipcfunModule);
    import_array();
    return module;
}
