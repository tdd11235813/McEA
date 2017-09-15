#include "catch.hpp"

#define N_RAD 2
#define POP_WIDTH 10

#include "../util/neighbor.cu"

TEST_CASE( "neighbors are calculated correctly (POP_WIDTH 10)", "[neighbor]" ) {
  REQUIRE( get_neighbor(5, 5, 0) == 36 );
  REQUIRE( get_neighbor(5, 5, 4) == 40 );
  REQUIRE( get_neighbor(5, 5, 12) == 60 );
  REQUIRE( get_neighbor(5, 5, 20) == 80 );
  REQUIRE( get_neighbor(5, 5, 24) == 84 );
}
