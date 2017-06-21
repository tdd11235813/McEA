#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))
#define W 7
#define NRAD 2

int *get_neighbor(int par_i, int par_j, int n_index, int *n_location) {

  // height and width of parent column and row
  int par_l = W - par_j;
  int par_k = W - par_i;
  //printf("parl: %d, park: %d\n", par_l, par_k);

  // differential indices
  int n_width = 2 * NRAD + 1;
  int d_i = n_index / n_width - NRAD;
  int d_j = n_index % n_width - NRAD;
  // printf("d_i: %d, d_j: %d\n", d_i, d_j);

  // height and width of neighbor column and row
  int n_l = MAX(MIN(par_l - d_j, W), 1);
  int n_k = MAX(MIN(par_k - d_i, W), 1);
  // printf("n_l: %d, n_k: %d\n", n_l, n_k);

  // prevent double negative index error
  int dneg = (par_i + d_i < 0 && par_j + d_j < 0);
  n_l = (dneg)? (W+1)/2:n_l;
  n_k = (dneg)? (W+1)/2:n_k;

  // calculate neighbor indices
  int n_i = (par_i + d_i + n_l*NRAD) % n_l; // + n_l*RAD prevents negative results
  int n_j = (par_j + d_j + n_k*NRAD) % n_k; // because: c modulo

  n_location[0] = n_i;
  n_location[1] = n_j;

  return n_location;
}

int *get_index_2d(int index_1d, int *index_2d) {

  int num_cells = (W * (W+1))/2;
  int k = (int)(sqrt( 1 + 8 * (num_cells - index_1d) ) / 2 + 0.49999999) ;
  // original version:
  // int k = (int)( ceil( (sqrt( 1 + 8 * (num_cell - i) ) -1) / 2 ) );

  // row width
  int index_i = W - k;
  index_2d[0] = index_i;

  // calc start of row
  int index_0 = num_cells - (k * (k + 1)) / 2;
  index_2d[1] = index_1d - index_0;

  return index_2d;
}

int get_index_1d(int index_2d_i, int index_2d_j) {
  int l = W - index_2d_i;
  return (W * (W + 1)) / 2 - (l * (l + 1)) / 2 + index_2d_j;
}

int main(int argc, char const *argv[]) {

  // ### calculations ###

  // create the memory indices
  int num_cell = (W * (W+1))/2;
  int *index_2d = malloc( 2 * sizeof(int) );
  int index_1d;

  // ### test for correct results ###
  int test_i = 0, test_j = 0;
  int row_width = W;
  for (size_t i = 0; i < num_cell; i++) {
      index_2d = get_index_2d(i, index_2d);
      index_1d = get_index_1d(index_2d[0], index_2d[1]);

      if(test_i != index_2d[0] || test_j != index_2d[1])
        printf("wrong index: (%2d, %2d) should be (%2d, %2d)\n", index_2d[0], index_2d[1], test_i, test_j);

      if(i != index_1d)
        printf("wrong 1d index: (%2d, %2d) should convert to: %2ld, instead: %2d\n", index_2d[0], index_2d[1], i, index_1d);

      test_j++;
      if(test_j == row_width) {
        test_j = 0;
        test_i++;
        row_width --;
      }
  }

  // ### print the results ###
  printf("topology width:\t\t%d\n", W);
  printf("number of cells:\t%d\n", num_cell);

  // indices
  int last, curr_i, curr_j;
  for (size_t i = 0; i < num_cell; i++) {
    index_2d = get_index_2d(i, index_2d);
    curr_i = index_2d[0];
    curr_j = index_2d[1];
    if(last != curr_i)
      printf("\n");
    printf("(%2d,%2d) ", curr_i, curr_j);
    last = curr_i;
  }
  printf("\n");


  // test neighbor calculation
  int *neigh = malloc( 2*sizeof(int));
  int neighbor_count = (2*NRAD+1)*(2*NRAD+1);
  for (size_t i = 0; i < neighbor_count; i++) {
    if(i % (2*NRAD+1) == 0)
      printf("\n");
    neigh = get_neighbor( 0, 4, i, neigh );
    printf("(%2d, %2d), ", neigh[0], neigh[1]);
  }

  // cleanup
  free( neigh );
  free( index_2d );

  return 0;
}
