#include "catch.hpp"
#include <cmath>

#define DTLZ_NUM 4

#include "../util/dtlz.cu"

TEST_CASE("dtlz4 calculates right", "[dtlz]") {
  int p_size = 7;
  int o_size = 3;
  float obj[3];

  float params0[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; 
  dtlz( params0, obj, p_size, o_size, 1 );
  REQUIRE( obj[0] == Approx( 2.25 ) );
  REQUIRE( obj[1] == Approx( 0.0 ) );
  REQUIRE( obj[2] == Approx( 0.0 ) );

  float params1[] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; 
  dtlz( params1, obj, p_size, o_size, 1 );
  REQUIRE( obj[0] == Approx( 0.0 ) );
  REQUIRE( obj[1] == Approx( 0.0 ) );
  REQUIRE( obj[2] == Approx( 2.25 ) );

  float params05[] = { 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}; 
  dtlz( params05, obj, p_size, o_size, 1 );
  REQUIRE( obj[0] == Approx( 1.0 ) );
  REQUIRE( obj[1] == Approx( 0.0 ) );
  REQUIRE( obj[2] == Approx( 0.0 ) );
}
