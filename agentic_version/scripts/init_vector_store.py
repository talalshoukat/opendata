#!/usr/bin/env python3
"""
Initialize vector store with categorical data from database
"""

import sys
import os

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



def initialize_vector_store():
    """Initialize vector store with database data"""
    print("🚀 Initializing Vector Store with Database Data")
    print("=" * 50)
    
    try:
        from tools.database_manager import DatabaseManager
        from tools.vector_store import FAISSVectorStore
        from config.config import Config
        
        # Create database manager
        print("📊 Connecting to database...")
        db_manager = DatabaseManager()
        
        # Get all schemas
        print("📋 Retrieving database schemas...")
        schemas = db_manager.get_all_schemas()
        print(f"✅ Retrieved {len(schemas)} table schemas")
        
        # Create vector store
        print("🔍 Creating vector store...")
        vector_store = FAISSVectorStore()
        
        # Add categorical values from database
        total_keywords = 0
        for table_name, schema in schemas.items():
            print(f"\n📝 Processing table: {table_name}")
            
            # Get actual data from the table
            sample_data = db_manager.get_sample_data(table_name, limit=100)
            
            if not sample_data.empty:
                # Extract only categorical columns (exclude numeric, ID, and date columns)
                categorical_data = {}
                for column in sample_data.columns:
                    # Skip columns that are likely not categorical
                    if any(skip in column.lower() for skip in ['id', 'count', 'number', 'amount', 'percentage', 'ratio', 'total', 'sum', 'avg', 'min', 'max']):
                        continue
                    
                    # Skip columns that are mostly numeric
                    numeric_count = 0
                    total_count = 0
                    unique_values = sample_data[column].dropna().unique()
                    
                    for val in unique_values:
                        total_count += 1
                        try:
                            float(str(val))
                            numeric_count += 1
                        except ValueError:
                            pass
                    
                    # Only include columns that are less than 50% numeric and have reasonable number of unique values
                    if total_count > 0 and (numeric_count / total_count) < 0.5 and len(unique_values) > 1 and len(unique_values) < 100:
                        # Convert to strings and filter out empty values
                        string_values = [str(val).strip() for val in unique_values if str(val).strip()]
                        if string_values:
                            categorical_data[column] = string_values
                
                if categorical_data:
                    # Add to vector store
                    vector_store.add_categorical_values(table_name, categorical_data)
                    
                    # Count keywords
                    table_keywords = sum(len(values) for values in categorical_data.values())
                    total_keywords += table_keywords
                    
                    print(f"   ✅ Added {len(categorical_data)} categorical columns with {table_keywords} unique values")
                    
                    # Show some examples
                    for column, values in categorical_data.items():
                        print(f"      - {column}: {len(values)} values (e.g., {', '.join(values[:3])}{'...' if len(values) > 3 else ''})")
                else:
                    print(f"   ⚠️  No categorical data extracted from {table_name}")
                    
                # Show what was excluded
                excluded_columns = [col for col in sample_data.columns if col not in categorical_data]
                if excluded_columns:
                    print(f"   ⚠️  Excluded columns (not categorical): {', '.join(excluded_columns[:5])}{'...' if len(excluded_columns) > 5 else ''}")
            else:
                print(f"   ⚠️  No data found in {table_name}")
        
        # Build index
        print(f"\n🔨 Building vector index...")
        vector_store.build_index()
        print(f"✅ Built vector index with {len(vector_store.keywords)} keywords")
        
        # Save the vector store
        print(f"💾 Saving vector store to {Config.VECTOR_STORE_PATH}...")
        vector_store.save_store()
        print("✅ Vector store saved successfully")
        
        # Test the vector store
        print(f"\n🧪 Testing vector store functionality...")
        test_queries = [
            "Show me technology contributors",
            "What are the top legal entities?",
            "How many engineers are there?",
            "Show me healthcare data",
            "What about finance sector?"
        ]
        
        for query in test_queries:
            result = vector_store.normalize_query_keywords(query)
            print(f"Query: {query}")
            print(f"Normalized: {result['normalized_query']}")
            print(f"Replacements: {len(result['replacements'])}")
            if result['replacements']:
                for replacement in result['replacements']:
                    print(f"  - '{replacement['original_keyword']}' → '{replacement['replaced_with']}' (confidence: {replacement['confidence']:.2f})")
            print()
        
        # Show statistics
        stats = vector_store.get_collection_stats()
        print(f"📊 Vector Store Statistics:")
        print(f"   Total keywords: {stats['total_keywords']}")
        print(f"   Tables processed: {stats['tables_processed']}")
        print(f"   Index size: {stats['index_size']}")
        print(f"   Is fitted: {stats['is_fitted']}")
        
        # Clean up
        db_manager.close()
        print("\n✅ Database connections closed")
        print("🎉 Vector store initialization completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector store initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🎯 Vector Store Initialization")
    print("This script populates the vector store with categorical data from the database")
    
    success = initialize_vector_store()
    
    if success:
        print("\n✨ Vector store is ready for use!")
    else:
        print("\n❌ Vector store initialization failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
