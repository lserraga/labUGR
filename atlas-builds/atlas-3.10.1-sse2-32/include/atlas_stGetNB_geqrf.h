#ifndef ATL_stGetNB_geqrf

/*
 * NB selection for GEQRF: Side='RIGHT', Uplo='UPPER'
 * M : 25,180,420,900,1860,3780
 * N : 25,180,420,900,1860,3780
 * NB : 12,60,60,60,60,120
 */
#define ATL_stGetNB_geqrf(n_, nb_) \
{ \
   if ((n_) < 102) (nb_) = 12; \
   else if ((n_) < 2820) (nb_) = 60; \
   else (nb_) = 120; \
}


#endif    /* end ifndef ATL_stGetNB_geqrf */
